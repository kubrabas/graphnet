# P-ONE Muon-CC Joint Direction Pipeline

Bu dizin mevcut `08_pone` pipeline'ına dokunmadan çalışan yeni ve paralel
direction pipeline'ıdır. Router yoktur, classification modeli yoktur ve zenith
ile azimuth için ayrı modeller yoktur. Tek DynEdge modeli bir 3B yön vektörü ve
bir `kappa` üretir.

Kaynak parquet dosyaları salt okunur kullanılır. Pipeline parquetlere weight
kolonu eklemez, onları kopyalamaz, yeniden bölmez veya silmez.

SLURM ortamı mevcut `/home/kbas/SlurmScripts/GraphNet` scriptleriyle aynı
şekildedir: `rorsoe/graphnet:graphnet-1.8.0-...` containerı CUDA, PyTorch ve
diğer bağımlılıkları sağlar; kullanılan GraphNeT Python kodu ise local
`/project/def-nahee/kbas/graphnet/src` ağacıdır. Her train/inference job'u
`graphnet.__file__` yolunu başlangıçta doğrular ve local kaynak yerine
containerdaki paket import edilmişse training başlamadan durur. Gerçek import
yolu `run_manifest.json` ve `inference_manifest.json` içine de yazılır.

## Kesinleşen kapsam

- Dataset: `340StringMC`
- Event seçimi: yalnız Muon CC
- Pulse seçimi: ilgili geometry'nin `triggered_nonoise_* == 1` eventleri
- Model hedefi: joint `(x, y, z, kappa)` direction
- Stage A loss: energy-weighted 3B vMF
- Stage B loss: energy-weighted `opening_angle + 0.05 * vMF`
- Fizik sonucu: median opening angle; global ve true-energy binlerinde
- Ana validation checkpoint adayı: energy-bin macro median opening angle
- Çıktı kökü:
  `Results/340StringMC_JointDirection/<geometry>/<experiment>/`

## Mevcut parquetler kullanılabilir mi?

Evet. Yapılan read-only denetimin sonucu:

| Geometry | Train | Validation | Test |
|---|---:|---:|---:|
| 102 string | 164,226 | 20,525 | 20,536 |
| 160 string | 187,077 | 23,378 | 23,400 |
| Full/340 | 335,222 | 41,893 | 41,927 |

Her geometry için:

- train, validation ve test fiziksel event bazında tamamen ayrık;
- fiziksel anahtar `(RunID, SubrunID, EventID, SubEventID)` benzersiz;
- `abs(pid) == 14`, `is_CC == 1`, `category1_isMuonCC == 1`;
- ilgili `triggered_nonoise_*` alanı bütün eventlerde `1`;
- `totalEnergy`, `zenith`, `azimuth` ve beş model feature'ı mevcut;
- enerji yaklaşık `100 GeV` ile `1 PeV` arasında;
- train-only RobustScaler percentile CSV'leri hazır.

Bu koşullar training başlamadan önce `data_audit.json` üreten kod tarafından
yeniden kontrol edilir. Bir koşul bozulmuşsa job training'e geçmeden durur.
Geometry'ler arası event eşleştirmesinde `event_no` kullanılmamalıdır;
fiziksel dört alan kullanılmalıdır.

Mevcut ayrı zenith+azimuth validation predictionları fiziksel ID ile
birleştirildiğinde başlangıç referansı şöyledir:

| Geometry | Mean angle | Median angle | q68 | `<1°` |
|---|---:|---:|---:|---:|
| 102 | 12.066° | 6.047° | 9.859° | 3.70% |
| 160 | 9.420° | 4.378° | 7.502° | 7.17% |
| Full | 9.638° | 4.567° | 7.555° | 6.05% |

Yeni pipeline'ın ilk görevi bu reference'ı aynı validation splitinde geçmektir.
`1°` çizgisi raporlarda gösterilir ama loss'u 1° altında kapatan bir threshold
değildir.

## Joint direction nasıl çalışıyor?

Head üç ham sayı üretir. Bunların uzunluğu `kappa`, normalize edilmiş hali ise
tahmin edilen yön olur:

```text
raw = (a, b, c)
kappa = ||raw||
predicted_direction = raw / kappa
```

Dolayısıyla zenith ve azimuth aynı fiziksel yönün iki bağımsız tahmini değildir.
Inference sırasında yön vektörü tekrar zenith ve azimuth'a çevrilir.

Target listesinde `[zenith, azimuth, totalEnergy]` bulunur. `totalEnergy`
backbone'a input olarak verilmez; yalnızca loss weight lookup'ında kullanılır.

## Energy weighting

Weight histogramı yalnız train truth dosyalarından fit edilir:

```text
log10(E/GeV) binleri: 2.0 -> 6.0
bin genişliği: 0.1 dex
raw weight: N_bin ** (-alpha)
normalizasyon: train event ortalaması = 1
clipping: [0.2, 5.0]
```

İlk pilot configlerinde `alpha=0.5` bulunur. Bu değer için notebook sonuçları:

| Geometry | Min weight | Median | Max | Effective sample fraction |
|---|---:|---:|---:|---:|
| 102 | 0.646 | 0.860 | 3.305 | 0.835 |
| 160 | 0.611 | 0.856 | 3.525 | 0.817 |
| Full | 0.586 | 0.833 | 3.886 | 0.791 |

Bu pilotta `[0.2, 5.0]` clipping fiilen hiçbir eventi değiştirmiyor; guard olarak
configte kalıyor. Clip, son `train mean = 1` normalizasyonundan önce uygulanır;
ileride daha agresif `alpha` kullanılırsa final sınırların ayrıca kontrol edilmesi
gerekir. Fit edilen edge, count ve weightler
`energy_weight_manifest.json` içine yazılır ve checkpoint buffer'larında da
saklanır.

Native GraphNeT parquet `loss_weight` yolu kullanılmaz. Mevcut sürümde event
loss'u `[N]`, native weight ise `[N,1]` olduğundan çarpım `[N,N]` broadcast
edebilir. Buradaki helper hem loss hem weight shape'ini tam `[N]` olarak kontrol
eder ve batch loss'unu gerçekten şöyle hesaplar:

```text
sum_i(weight_i * loss_i) / sum_i(weight_i)
```

## İki training aşaması

Stage A:

```text
L_i = vMF_i
```

Stage B:

```text
L_i = opening_angle_i [radian] + 0.05 * vMF_i
```

İlk configler Kaggle ikinci çözümündeki geçişi takip ederek Stage A'yı 3 epoch
çalıştırır. Stage B daha düşük learning rate ile en fazla 30 epoch sürer ve
primary median metriğinde 7 epoch iyileşme olmazsa durur. Bunların hepsi YAML
içinde değiştirilebilir.

Median doğrudan training loss değildir. Median sıralama işlemidir ve batchteki
eventlerin çoğuna yararlı gradient vermez. Differentiable vMF/hybrid loss modele
öğretir; median opening angle validation başarısını ölçer.

## Validation ve checkpointler

Opening-angle raporu için varsayılan true-energy binleri:

```text
log10(E/GeV) = [2.0, 2.5, 3.0, ..., 5.5, 6.0]
```

`val_macro_median_deg`, bu sekiz binin median opening angle değerlerinin eşit
ortalamasıdır. Böylece düşük enerjide daha fazla event bulunması diğer enerji
bölgelerini görünmez yapmaz.

Her epoch şu değerler kaydedilir:

- global mean, median, q68, q84 ve q90 opening angle;
- `<1°`, `<2°`, `<5°` ve `<10°` event oranları;
- aynı değerlerin her true-energy binindeki sonucu;
- macro mean, median, q68, q84 ve q90;
- weighted ve unweighted vMF, hybrid ve aktif objective loss;
- kappa özeti, learning rate, epoch süresi, RAM ve GPU snapshotları.

Validation normal event dağılımını değiştirmez. Weighted validation loss ayrıca
şu diagnostik olarak hesaplanır:

```text
sum_i(train-derived weight(E_i) * validation loss_i) / sum_i(weight(E_i))
```

Bu değer primary physics metriği değildir. Kappa iyileşirken opening angle aynı
kalabileceği için yalnız weighted training objective'in validation'a taşınıp
taşınmadığını gösterir.

Tek run aşağıdaki resumable `.ckpt` ve sade `.pth` dosyalarını ayrı saklar:

- `best_macro_median`
- `best_global_median`
- `best_global_q68`
- `best_weighted_objective`
- `last`

Primary early stopping monitor şu anda `val_macro_median_deg` değeridir. Bütün
metrikler CSV'ye yazıldığı ve ayrı checkpointler korunduğu için daha sonra
global median kararına dönmek modelin kaybolmasına yol açmaz.

Training preflight test splitinde yalnız schema, pulse coverage, fiziksel ID,
Muon-CC/trigger bayrakları ve split overlap kontrolü yapar. Test `zenith`,
`azimuth` ve `totalEnergy` değerlerini okumaz; test loader'ı da oluşturmaz.
Validation ile checkpoint dondurulana kadar test target dağılımı kapalı kalır.

## Model derinliği

İlk deney mevcut pipeline ile adil karşılaştırma için aynı standart DynEdge
backbone'unu kullanır. Configteki şu üç alan daha sonraki kontrollü depth/width
ablation için hazırdır:

```yaml
dynedge_layer_sizes: null
post_processing_layer_sizes: null
readout_layer_sizes: null
```

Önce joint target + doğru loss + weighting etkisi ölçülmelidir. Derin model aynı
anda açılırsa iyileşmenin hangi değişiklikten geldiği anlaşılamaz. Yeni mimari
denenirken `experiment_name` mutlaka değiştirilmelidir.

## Configler

```text
configs/102_string_emax1e6.yml
configs/160_string_emax1e6.yml
configs/full_geometry_emax1e6.yml
```

Her config yalnız kendi canonical
`STRING340MC_PARQUET[geometry]["Muon"][train|val|test]` yollarını açar. Scaler
dosya yolunda tarihsel `category1_isMuonCC/class_1_muon_cc` adı geçse de router
çalıştırılmaz; bu CSV aynı Muon-CC train featurelarından önceden hesaplanmış
P25/P50/P75 değerleridir.

## Training gönderme

Önce komutu SLURM'a göndermeden görmek için:

```bash
cd /project/def-nahee/kbas/graphnet
python3 examples/09_pone_muon_direction/slurm/submit_train.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --dry-run
```

Gerçek 102-string training:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_train.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml
```

`--stage all` varsayılandır: Stage A biter, `last.ckpt` Stage B'yi initialize
eder ve aynı job devam eder. Stage A daha önce tamamlandıysa Stage B tek başına:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_train.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --stage stage_b
```

Kesilmiş Stage B job'unu optimizer ve scheduler dahil resume etmek için:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_train.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --stage stage_b --resume
```

Pipeline otomatik silme/overwrite seçeneği sunmaz. Yeni bilimsel deney için yeni
bir `experiment_name` kullanılmalıdır.

## 102-string weights-only fine-tuning

İlk `joint_weighted_alpha05_v1` Stage-B run'ı 30 epoch sınırına ulaştığında
validation macro median hâlâ iyileşiyordu. Bu nedenle onun primary checkpointi
ayrı bir deneyde daha küçük bir learning-rate restart ile devam ettirilebilir.
Kaynak model sabitlenmiştir:

```text
experiment: joint_weighted_alpha05_v1
checkpoint: stage_b_angular_hybrid/checkpoints/best_macro_median.ckpt
source epoch: 29
source val_macro_median_deg: 5.6466331482
SHA256: 5f2e4007abbf69e9e685ad74263a55698150fc95ea9950bdd03866cf028fbe01
```

Fine-tune configi:

```text
configs/102_string_emax1e6_finetune_lr1e5.yml
```

Bu akış mevcut run üzerinde `--resume` yapmaz. Kaynak checkpointten yalnızca
model `state_dict` değerlerini `strict=True` ile yükler; eski Adam, LR scheduler,
epoch ve early-stopping durumunu yüklemez. Yeni optimizer `1e-6` ile başlar,
yarım epoch içinde `1e-5` değerine çıkar ve en fazla 50 yeni epoch boyunca
`1e-6` değerine iner. Primary macro median yedi epoch iyileşmezse early stopping
daha önce durdurur.

Yeni çıktı tamamen ayrı bir sibling dizindedir:

```text
Results/340StringMC_JointDirection/102_string_emax1e6/
  joint_weighted_alpha05_v1_finetune_lr1e5/
```

Kaynak checkpoint SHA256'sı, checkpoint index kaydı, source resolved config ve
immutable data/weight/loss/model ayarları job başlamadan doğrulanır. Yeni train
manifesti de kaynak energy-weight manifestiyle birebir aynı olmak zorundadır.
Provenance yeni output içindeki `finetune_source_manifest.json` dosyasına yazılır;
kaynak experimentte hiçbir dosya oluşturulmaz veya değiştirilmez.

Önce salt-okunur dry-run:

```bash
cd /project/def-nahee/kbas/graphnet
python3 examples/09_pone_muon_direction/slurm/submit_finetune.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6_finetune_lr1e5.yml \
  --dry-run
```

Gerçek gönderim:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_finetune.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6_finetune_lr1e5.yml
```

Yalnız yeni fine-tune job'ı gerçekten kesilmiş ve kendi `last.ckpt` dosyasını
üretmişse optimizer/scheduler/patience dahil güvenli resume yapılabilir:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_finetune.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6_finetune_lr1e5.yml \
  --resume
```

Yeni experimentteki `best_macro_median`, yalnız fine-tune epochları arasındaki
en iyidir. Fine-tune sonucunun gerçekten kazanç sayılması için kaynak değer
`5.646633°` ile ayrıca karşılaştırılmalıdır. Kaynak checkpoint daima korunmuş
fallback modelidir. Test splitine bakma/freeze kuralları aynen geçerlidir.

## Validation ve test inference

Önce iki median checkpointi validation üzerinde raporlamak güvenli akıştır:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_inference.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --split val --checkpoint-name best_macro_median

python3 examples/09_pone_muon_direction/slurm/submit_inference.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --split val --checkpoint-name best_global_median
```

İki validation job'u tamamlandıktan ve CSV sonuçları karşılaştırıldıktan sonra
tek checkpoint seçimi açık bir validation gerekçesiyle dondurulur:

```bash
python3 examples/09_pone_muon_direction/freeze_checkpoint.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --stage stage_b \
  --checkpoint-name best_macro_median \
  --reason "Validation macro median daha düşük"
```

Bu komut checkpoint SHA256 değerini ve validation metriklerini
`checkpoint_selection.json` içine yazar. Daha önce test artifactı varsa veya
seçim zaten dondurulmuşsa dosyayı değiştirmeden durur. Test inference yalnız bu
dondurulmuş checkpoint ile ve yalnız bir kez çalıştırılmalıdır:

```bash
python3 examples/09_pone_muon_direction/slurm/submit_inference.py \
  -c examples/09_pone_muon_direction/configs/102_string_emax1e6.yml \
  --split test --checkpoint-name best_macro_median
```

Inference çıktısında fiziksel ID'ler, true/predicted xyz, zenith, azimuth,
kappa, opening angle, true energy ve train-derived diagnostik weight bulunur.

## Çıktı düzeni

```text
Results/340StringMC_JointDirection/<geometry>/<experiment>/
├── pipeline_config.yml
├── resolved_config.yml
├── run_manifest.json
├── finetune_source_manifest.json  # yalnız fine-tune experimentlerinde
├── data_audit.json
├── energy_weight_manifest.json
├── checkpoint_selection.json
├── stage_a_vmf/
│   ├── training_history_by_epoch.csv
│   ├── validation_metrics_by_energy_epoch.csv
│   ├── resources_and_time.csv
│   ├── training_validation_loss.png
│   ├── validation_opening_angle_by_epoch.png
│   └── checkpoints/
├── stage_b_angular_hybrid/
│   └── ... aynı training/checkpoint dosyaları ...
└── inference/
    ├── val/<stage_checkpoint>/
    └── test/<stage_checkpoint>/
        ├── predictions.parquet
        ├── metrics_summary.csv
        ├── metrics_by_true_energy.csv
        ├── opening_angle_distribution.png
        ├── opening_angle_cdf.png
        └── opening_angle_by_true_energy.png
```

## Testler

Core helper testleri GraphNeT containerı olmadan da çalışır:

```bash
cd /project/def-nahee/kbas/graphnet/examples/09_pone_muon_direction
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

Testler direction dönüşümleri, 0°/90°/180° opening angle, vMF gradienti,
weight fit/clip/normalize, manifest round-trip, checkpoint bufferları, broadcast
guard, gerçek `opening_angle + 0.05*vMF` backward yolu, `sum(wL)/sum(w)` ve
global/macro median ayrımını kapsar.

## Referans yaklaşım

İki aşamalı vMF -> `opening angle + 0.05*vMF` fikri
[IceCube Kaggle top-3 çözüm makalesindeki](https://link.springer.com/article/10.1140/epjc/s10052-024-12977-2)
ikinci çözümden gelir. Buradaki energy weighting, Muon-CC seçimi, train-only
manifest, macro-median monitoring ve güvenli GraphNeT entegrasyonu bu P-ONE
dataseti için eklenmiştir.
