"""Test One Hot Encoding."""
import pytest

from fe_polars.encoding.one_hot_encoding import OneHotEncoder


def test_one_hot_encoder(standard_polars_dataframe):
    """Test one hot encoding.

    - Assert that the output has the correct number of columns
    - Assert that the correct columns are created
    """
    encoder_keep = OneHotEncoder(features_to_encode="City", strategy="keep")
    encoder_drop = OneHotEncoder(features_to_encode="City", strategy="drop")
    keep = encoder_keep.fit_transform(standard_polars_dataframe)
    drop = encoder_drop.fit_transform(standard_polars_dataframe)

    assert len(keep.columns) == 6
    assert len(drop.columns) == 5
    for onehot in ["City_A", "City_B", "City_C"]:
        assert onehot in keep.columns
    for onehot in ["City_A", "City_B", "City_C"]:
        assert onehot in drop.columns


def test_features_to_encode_type():
    """Test if using str or list for features_to_encode works correctly."""
    str_encoder = OneHotEncoder(features_to_encode="City")
    list_encoder = OneHotEncoder(features_to_encode=["City"])

    assert isinstance(str_encoder.features_to_encode, list)
    assert isinstance(list_encoder.features_to_encode, list)


def test_bad_strategy():
    """Test bad strategy.

    - Assert a wrong strategy parameter is correctly handled.
    """
    with pytest.raises(ValueError) as excinfo:
        _ = OneHotEncoder(features_to_encode=["City"], strategy="bad-strategy")
    assert str(excinfo.value) == "strategy must be one of ['keep', 'drop']"
