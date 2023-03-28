"""Fixtures for unit tests."""
import polars
import pytest


@pytest.fixture
def standard_polars_dataframe():
    """Fixture for a standard polars dataframe.

    Returns:
        polars.DataFrame: polars dataframe
    """
    return polars.DataFrame(
        {
            "City": ["A", "A", "B", "B", "B", "C", "C", "C"],
            "Temperature": [30.5, 32, 25, 38, 40, 29.6, 21.3, 24.9],
            "Rain": [103, 125, 90, 75, 130, 200, 155, 127],
        }
    )


@pytest.fixture
def with_categorical_nulls_polars_dataframe():
    """Fixture for a standard polars dataframe.

    Returns:
        polars.DataFrame: polars dataframe
    """
    return polars.DataFrame(
        {
            "City": ["A", "A", "B", "B", None, "C", "C", None],
            "Temperature": [30.5, 32, 25, 38, 40, 29.6, 21.3, 24.9],
            "Rain": [103, 125, 90, 75, 130, 200, 155, 127],
        }
    )


@pytest.fixture
def with_numerical_nulls_polars_dataframe():
    """Fixture for a standard polars dataframe.

    Returns:
        polars.DataFrame: polars dataframe
    """
    return polars.DataFrame(
        {
            "City": ["A", "A", "B", "B", "B", "C", "C", "C"],
            "Temperature": [30.5, 32, 25, 38, 40, 29.6, 21.3, 24.9],
            "Rain": [103, None, 90, 75, None, 200, 155, 127],
        }
    )


@pytest.fixture
def standard_polars_series():
    """Fixture for a standard polars series.

    Returns:
        polars.Series: polars series
    """
    return polars.Series(
        "Rain", [103, 125, 90, 75, 130, 200, 155, 127], dtype=polars.Float32
    )
