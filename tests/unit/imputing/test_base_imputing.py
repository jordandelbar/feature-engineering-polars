"""Test Imputing.

The dataframe is as follows:
| City | Temperature | Rain   |
-------------------------------
| A    |    30.5     |  103   |
| A    |    32       |  None  |
| B    |    32       |  90    |
| B    |    38       |  75    |
| B    |    40       |  None  |
| C    |    29.6     |  200   |
| C    |    21.3     |  155   |
| C    |    None     |  127   |

For the column `Rain`:
-> The global mean to be imputed should be: 125
-> The global median to be imputed should be: 115
-> Since there are two `None` in our dataframe the imputer should default
to the second choice in terms of frequence (200 here since we sort by counts
then values).
-> The max value to be imputed should be: 200
-> The min value to be imputed should be: 75
"""
import math

import polars
import pytest

from fe_polars.imputing.base_imputing import Imputer


def test_mean_imputing(with_numerical_nulls_polars_dataframe):
    """Test mean imputing.

    - Assert if mean imputing is correctly done
    - Assert that there is no more null values
    """
    imputer = Imputer(features_to_impute=["Rain"], strategy="mean")
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 125, abs_tol=0.001)


def test_median_imputing(with_numerical_nulls_polars_dataframe):
    """Test median imputing.

    - Assert if median imputing is correctly done
    - Assert that there is no more null values after transform
    """
    imputer = Imputer(features_to_impute=["Rain"], strategy="median")
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    result = result.select("Rain")[1, :].item()

    assert math.isclose(result, 115, abs_tol=0.001)


def test_mode_imputing(
    with_categorical_nulls_polars_dataframe, with_numerical_nulls_polars_dataframe
):
    """Test median imputing.

    - Assert if mode imputing is correctly done
    - Assert every polars dtype is correctly handled with Exception
    - Assert that there is no more null values after transform
    """
    # Those dtypes should pass
    for dtype in [
        polars.UInt16,
        polars.UInt32,
        polars.UInt64,
        polars.UInt8,
        polars.Utf8,
        polars.Int16,
        polars.Int32,
        polars.Date,
        polars.Datetime,
        polars.Binary,
        polars.Duration,
        polars.Time,
    ]:
        imputer_mt = Imputer(features_to_impute=["Temperature"], strategy="mode")
        result_mt = imputer_mt.fit_transform(
            with_categorical_nulls_polars_dataframe.with_columns(
                polars.col("Temperature").cast(dtype=dtype)
            )
        )
        assert result_mt["Temperature"].null_count() == 0
        if dtype == polars.Utf8:
            assert result_mt["Temperature"][7] == "32.0"

    # Those should raise an exception
    for dtype in [
        polars.Float32,
        polars.Float64,
        polars.Decimal,
        polars.Boolean,
    ]:
        imputer_mt = Imputer(features_to_impute=["Temperature"], strategy="mode")
        with pytest.raises(TypeError) as excinfo:
            _ = imputer_mt.fit_transform(
                with_categorical_nulls_polars_dataframe.with_columns(
                    polars.col("Temperature").cast(dtype=dtype)
                )
            )
            assert (
                str(excinfo.value)
                == f"dtype `{dtype}` is not supported for mode strategy"
            )
    # Second choice test
    imputer_sct = Imputer(features_to_impute=["Rain"], strategy="mode")
    result_sct = imputer_sct.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result_sct["Rain"].null_count() == 0

    result_sct = result_sct.select("Rain")[1, :].item()
    assert math.isclose(result_sct, 200, abs_tol=0.001)

    # Multiple most-frequent values
    imputer_mmf = Imputer(features_to_impute=["City"], strategy="mode")
    result_mmf = imputer_mmf.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result_mmf["City"].null_count() == 0


def test_min_max_imputing(with_numerical_nulls_polars_dataframe):
    """Test min and max imputing.

    - Assert if min and max imputing are correctly done
    - Assert that there is no more null values after transform
    """
    max_imputer = Imputer(features_to_impute=["Rain"], strategy="max")
    min_imputer = Imputer(features_to_impute=["Rain"], strategy="min")
    result_max = max_imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result_max["Rain"].null_count() == 0

    result_min = min_imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result_min["Rain"].null_count() == 0

    result_max = result_max.select("Rain")[1, :].item()
    result_min = result_min.select("Rain")[1, :].item()

    assert math.isclose(result_max, 200, abs_tol=0.001)
    assert math.isclose(result_min, 75, abs_tol=0.001)


def test_strategy_dict(with_numerical_nulls_polars_dataframe):
    """Test the strategy dict option."""
    imputer = Imputer(strategy_dict={"Rain": "max"})
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result["Rain"].null_count() == 0


def test_no_arg_provided(with_numerical_nulls_polars_dataframe):
    """Test the class if no argument is provided."""
    imputer = Imputer()
    result = imputer.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result["Rain"].null_count() == 0


def test_only_one_arg_provided(with_numerical_nulls_polars_dataframe):
    """Test the class if only one of the argument is provided."""
    imputer_strat = Imputer(strategy="mode")
    result_strat = imputer_strat.fit_transform(with_numerical_nulls_polars_dataframe)

    imputer_fti = Imputer(features_to_impute=["Rain"])
    result_fti = imputer_fti.fit_transform(with_numerical_nulls_polars_dataframe)
    assert result_strat["Rain"].null_count() == 0
    assert result_fti["Rain"].null_count() == 0


def test_bad_strategy():
    """Test bad strategy.

    - Assert a wrong strategy parameter is correctly handled.
    """
    with pytest.raises(ValueError) as excinfo:
        _ = Imputer(features_to_impute=["Rain"], strategy="bad-strategy")
        assert str(excinfo.value).contains("strategy must be one of")


def test_features_to_impute_type():
    """Test if using str or list for features_to_impute works correctly."""
    str_imputer = Imputer(features_to_impute="City")
    list_imputer = Imputer(features_to_impute=["City"])

    assert isinstance(str_imputer.features_to_impute, list)
    assert isinstance(list_imputer.features_to_impute, list)


def test_bad_parameter():
    """Test error message appears when a wrong parameter is provided."""
    with pytest.raises(ValueError) as excinfo:
        _ = Imputer(wrong_argument=["Rain"])
        assert str(excinfo.value).contains(
            "Invalid parameter 'wrong_argument' provided."
        )
