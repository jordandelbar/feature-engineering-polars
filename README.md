# Feature Engineering with Polars

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/feature-engineering-polars?logo=Python)](https://pypi.org/project/feature-engineering-polars/)
[![GitHub](https://img.shields.io/github/license/jordandelbar/feature-engineering-polars)](https://github.com/jordandelbar/feature-engineering-polars/blob/main/LICENSE.md)
[![codecov](https://codecov.io/gh/jordandelbar/feature-engineering-polars/branch/main/graph/badge.svg?token=TUKAPUHHEV)](https://codecov.io/gh/jordandelbar/feature-engineering-polars)

Feature engineering done with Polars

![fe-polars](https://user-images.githubusercontent.com/35341015/229273836-9f87fd67-2011-4aa9-a7d8-680795d75259.png)

## How to install

```bash
pip install feature-engineering-polars
```

## How to use it

```python
import polars as pl
from fe_polars.imputing.base_imputing import Imputer
from fe_polars.encoding.target_encoding import TargetEncoder

dataframe = pl.DataFrame(
        {
            "City": ["A", "A", "B", "B", "B", "C", "C", "C"],
            "Rain": [103, None, 90, 75, None, 200, 155, 127],
            "Temperature": [30.5, 32, 25, 38, 40, 29.6, 21.3, 24.9],
        }
    )

imputer = Imputer(features_to_impute=["Rain"], strategy="mean")
encoder = TargetEncoder(smoothing=2, features_to_encode=["City"])

temp = imputer.fit_transform(x=dataframe)
encoder.fit_transform(x=temp, y=dataframe['Temperature'])


shape: (8, 3)
City    Temperature Rain
f64     f64         f64

30.706  30.5        103.0
30.706  32.0        125.0
32.665  25.0        90.0
32.665  38.0        75.0
32.665  40.0        125.0
27.225  29.6        200.0
27.225  21.3        155.0
27.225  24.9        127.0
```

## Available transformers

- Encoding:
  - Target encoding
  - One hot encoding
- Imputing:
  - Base imputing:
    - Mean imputing
    - Median imputing
    - Max imputing
    - Min imputing
    - Fixed value imputing
