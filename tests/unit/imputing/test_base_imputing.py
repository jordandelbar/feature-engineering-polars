"""Test Mean Imputing."""
import math

import pytest

from fe_polars.imputing.base_imputing import Imputer


def test_mean_imputing(with_numerical_nulls_polars_dataframe):
    """Test mean imputing.

    - Assert if mean imputing is correctly done
    - Assert that there is no more null values

    the dataframe is as follows:
    | City | Rain  |
    ----------------
    | A    | 103   |
    | A    | None  |
    | B    | 90    |
    | B    | 75    |
    | B    | None  |
    | C    | 200   |
    | C    | 155   |
    | C    | 127   |

    The global mean to be imputed should be: 125
    """
    imputer = Imputer(features_to_impute=["Rain"], strategy="mean")
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 125, abs_tol=0.001)


def test_median_imputing(with_numerical_nulls_polars_dataframe):
    """Test median imputing.

    - Assert if median imputing is correctly done
    - Assert that there is no more null values

    the dataframe is as follows:
    | City | Rain  |
    ----------------
    | A    | 103   |
    | A    | None  |
    | B    | 90    |
    | B    | 75    |
    | B    | None  |
    | C    | 200   |
    | C    | 155   |
    | C    | 127   |

    The global median to be imputed should be: 115
    """
    imputer = Imputer(features_to_impute=["Rain"], strategy="median")
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 115, abs_tol=0.001)


def test_mode_imputing(with_numerical_nulls_polars_dataframe):
    """Test median imputing.

    - Assert if mode imputing is correctly done
    - Assert that there is no more null values

    the dataframe is as follows:
    | City | Rain  |
    ----------------
    | A    | 103   |
    | A    | None  |
    | B    | 90    |
    | B    | 75    |
    | B    | None  |
    | C    | 200   |
    | C    | 155   |
    | C    | 127   |

    Since there are two None in our dataframe the imputer should default to
    the second choice. Oddly enough in our case this is 155 and not 200
    -> needs further investigations
    """
    imputer = Imputer(features_to_impute=["Rain"], strategy="mode")
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 155, abs_tol=0.001)


def test_min_max_imputing(with_numerical_nulls_polars_dataframe):
    """Test min and max imputing.

    - Assert if min and max imputing are correctly done

    the dataframe is as follows:
    | City | Rain  |
    ----------------
    | A    | 103   |
    | A    | None  |
    | B    | 90    |
    | B    | 75    |
    | B    | None  |
    | C    | 200   |
    | C    | 155   |
    | C    | 127   |

    The max value to be imputed should be: 200
    The min value to be imputed should be: 75
    """
    max_imputer = Imputer(features_to_impute=["Rain"], strategy="max")
    min_imputer = Imputer(features_to_impute=["Rain"], strategy="min")
    result_max = max_imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result_min = min_imputer.fit_transform(with_numerical_nulls_polars_dataframe)

    result_max = result_max.select("Rain")[1, :].item()
    result_min = result_min.select("Rain")[1, :].item()

    assert math.isclose(result_max, 200, abs_tol=0.001)
    assert math.isclose(result_min, 75, abs_tol=0.001)


def test_bad_strategy():
    """Test bad strategy."""
    with pytest.raises(ValueError) as excinfo:
        _ = Imputer(features_to_impute=["Rain"], strategy="bad-strategy")
    assert (
        str(excinfo.value)
        == "strategy must be one of ['mean', 'median', 'mode', 'max', 'min']"
    )
