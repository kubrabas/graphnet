from typing import Dict, Callable, List, Optional

import pandas as pd
import torch

from graphnet.models.detector.detector import Detector


class PONE(Detector):
    """Generic P-ONE detector with RobustScaler normalization.

    All geometry and feature settings are instance-level so the same class
    works regardless of which coordinate system or feature set is used.

    Args:
        percentiles_csv:   Path to CSV with columns [feature, p25, p50, p75].
        xyz:               Column names for spatial coordinates used by
                           sensor_position_names (e.g. KNN graph building).
                           Defaults to ["pmt_x", "pmt_y", "pmt_z"].
        string_id_column:  Column name for string index. Defaults to "string".
        sensor_id_column:  Column name for sensor index. Defaults to "sensor_id".
        selected_features: If given, only these features are included in
                           feature_map(). Must be a subset of features present
                           in percentiles_csv.
        replace_with_identity: Features that skip normalization (passed through
                               as-is). See base Detector class.
        eps:               Minimum IQR threshold below which only median
                           subtraction is applied. Defaults to 1e-12.
    """

    def __init__(
        self,
        percentiles_csv: str,
        xyz: Optional[List[str]] = None,
        string_id_column: str = "string",
        sensor_id_column: str = "sensor_id",
        selected_features: Optional[List[str]] = None,
        replace_with_identity: Optional[List[str]] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(replace_with_identity=replace_with_identity)

        # Instance-level geometry settings (overrides any class-level defaults)
        self.xyz              = xyz if xyz is not None else ["pmt_x", "pmt_y", "pmt_z"]
        self.string_id_column = string_id_column
        self.sensor_id_column = sensor_id_column
        self._selected_features = selected_features
        self._eps = eps

        df = pd.read_csv(percentiles_csv)
        p = df.set_index("feature")
        self._p25 = p["p25"].to_dict()
        self._p50 = p["p50"].to_dict()
        self._p75 = p["p75"].to_dict()

    def _robust_scale(self, x: torch.Tensor, feature: str) -> torch.Tensor:
        if feature not in self._p50:
            raise KeyError(f"No percentiles for feature='{feature}'. Check percentiles_csv.")
        p25   = float(self._p25[feature])
        p50   = float(self._p50[feature])
        p75   = float(self._p75[feature])
        denom = p75 - p25
        x     = x.to(torch.float32)
        if abs(denom) < self._eps:
            return x - p50
        return (x - p50) / denom

    def feature_map(self) -> Dict[str, Callable]:
        """Build normalization map dynamically from percentiles CSV.

        If selected_features is set, only those features are included.
        Order follows selected_features (or percentiles_csv order if None).
        """
        features = list(self._p50.keys())
        if self._selected_features is not None:
            missing = [f for f in self._selected_features if f not in features]
            if missing:
                raise KeyError(f"Features not in percentiles_csv: {missing}")
            features = self._selected_features
        return {f: lambda x, f=f: self._robust_scale(x, f) for f in features}
