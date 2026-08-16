# Third-party notices

## IceCube second-place Fourier transformer

`fourier_spacetime_transformer.py` is a PyTorch-2.x P-ONE adaptation of ideas
and portions of the `DeepIceModel` implementation from:

- <https://github.com/DrHB/icecube-2nd-place>
- inspected revision `484cdcfed01af5255dce148122b095a7427ec1cb`
- Bukhari et al., *IceCube—Neutrinos in Deep Ice: the top 3 solutions from the
  public Kaggle competition*, EPJC 84 (2024) 646,
  <https://doi.org/10.1140/epjc/s10052-024-12977-2>

The upstream code is distributed under the following MIT license:

> MIT License
>
> Copyright (c) 2023 DrHB
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The adaptation does not vendor or install the upstream repository's historical
Python environment. In particular, it does not replace the production
GraphNeT container's PyTorch, PyG, pandas, polars, or CUDA packages.
