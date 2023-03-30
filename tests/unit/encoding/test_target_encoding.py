"""Test Target Encoding."""
import math

import polars

from feature_engineering_polars.encoding.target_encoding import TargetEncoder


def test_target_encoding(standard_polars_dataframe):
    """Test target encoding.

    - Assert that the provided dataframe is not modified.
    - Assert the computation is correct:

    the dataframe is as follows:
    | City | Rain |
    ---------------
    | A    | 103  |
    | A    | 125  |
    | B    | 90   |
    | B    | 75   |
    | B    | 130  |
    | C    | 200  |
    | C    | 155  |
    | C    | 127  |

    Global mean of `Rain` = 125.625

    According to the formula for smoothed target encoding,
    the computation for `City` A would be as follows:
    (2 * (103 + 125) / 2 + smoothing * 125.625) / (smoothing + 2)

    This gives the following values for different values of smoothing:
    with smoothing = 1 : 117.875
    with smoothing = 25 : 124.763
    """
    # We clone the dataframe to compare it at the end of the test
    sanity_check_dataframe = standard_polars_dataframe.clone()

    # Encoder with a smoothing set to 0
    encoder_smoothing_0 = TargetEncoder(smoothing=1, features_to_encode=["City"])
    encoded_dataframe_smoothing_0 = encoder_smoothing_0.fit_transform(
        x=standard_polars_dataframe.select(polars.col("City")),
        y=standard_polars_dataframe.select(polars.col("Rain")),
    )
    # Encoder with a smoothing set to 25
    encoder_smoothing_25 = TargetEncoder(smoothing=25, features_to_encode=["City"])
    encoded_dataframe_smoothing_25 = encoder_smoothing_25.fit_transform(
        x=standard_polars_dataframe.select(polars.col("City")),
        y=standard_polars_dataframe.select(polars.col("Rain")),
    )

    # Different encoding results for the `City` A
    smoothing_0_result = encoded_dataframe_smoothing_0[0, :].item()
    smoothing_25_result = encoded_dataframe_smoothing_25[0, :].item()

    # Assertions
    assert math.isclose(a=smoothing_0_result, b=117.875, abs_tol=0.001)
    assert math.isclose(a=smoothing_25_result, b=124.763, abs_tol=0.001)
    assert sanity_check_dataframe.frame_equal(standard_polars_dataframe)


def test_target_encoding_nulls(
    caplog, standard_polars_dataframe, with_categorical_nulls_polars_dataframe
):
    """Test target encoding with dataframe with null values.

    - Assert that the logger.warning is triggered correctly.
    """
    encoder = TargetEncoder(smoothing=1, features_to_encode=["City"])
    encoder.fit(
        x=standard_polars_dataframe.select(polars.col("City")),
        y=standard_polars_dataframe.select(polars.col("Rain")),
    )
    encoder.transform(
        x=with_categorical_nulls_polars_dataframe.select(polars.col("City"))
    )

    assert "['City'] have unseen values, defaults to global mean" in caplog.text


def test_target_encoding_with_series(standard_polars_dataframe, standard_polars_series):
    """Test target encoding with series in place of dataframe."""
    encoder = TargetEncoder(smoothing=1, features_to_encode=["City"])
    encoder.fit(
        x=standard_polars_dataframe.select(polars.col("City")), y=standard_polars_series
    )
    result = encoder.transform(standard_polars_dataframe.select(polars.col("City")))[
        0, :
    ].item()

    assert math.isclose(a=result, b=117.875, abs_tol=0.001)


def test_target_encoding_with_different_dtypes(caplog, standard_polars_dataframe):
    """Test if the data dtypes is correctly enforced."""
    dataframe_int = standard_polars_dataframe
    dataframe_str = standard_polars_dataframe.with_columns(
        polars.col("Rain").cast(polars.Utf8)
    )

    encoder = TargetEncoder(smoothing=1, features_to_encode=["Rain"])
    encoder.fit(x=dataframe_int.select("Rain"), y=dataframe_str.select("Temperature"))

    transformed_int = encoder.transform(x=dataframe_int)
    transformed_str = encoder.transform(x=dataframe_str)

    assert transformed_int.frame_equal(transformed_str)
    assert (
        "Feature ['Rain'] was mapped with dtype Int64 not Utf8, Int64 was enforced"
        in caplog.text
    )
