"""Test target encoding."""
import polars

from feature_engineering_polars.encoding.target_encoding import TargetEncoder


def test_target_encoding(standard_polars_dataframe):
    """Test target encoding."""
    sanity_check_dataframe = standard_polars_dataframe
    encoder = TargetEncoder(smoothing=1, features_to_encode=["City"])
    encoder.fit_transform(
        x=standard_polars_dataframe.select(polars.col("City")), y=standard_polars_dataframe.select(polars.col("Rain"))
    )
    assert sanity_check_dataframe.frame_equal(standard_polars_dataframe)


def test_target_encoding_nulls(with_nulls_polars_dataframe):
    """Test target encoding with dataframe with null values."""
    encoder = TargetEncoder(smoothing=1, features_to_encode=["City"])
    encoder.transform(x=with_nulls_polars_dataframe.select(polars.col("City")))
