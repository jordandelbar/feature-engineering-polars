"""Test Mean Imputing."""
import math

from fe_polars.imputing.mean_imputing import MeanImputer


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

    The blobal mean to be imputed should be: 125

    """
    imputer = MeanImputer(features_to_impute=["Rain"])
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 125, abs_tol=0.001)
