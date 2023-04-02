"""One hot encoding."""
from typing import List, Optional, Union

import polars


class OneHotEncoder:
    """One Hot Encoder class."""

    def __init__(self, features_to_encode: Union[str, List], strategy: str = "drop"):
        """Init.

        Args:
            features_to_encode (str | list): list of features to encode
            strategy (str): drop or keep the one hot encoded column
        """
        if isinstance(features_to_encode, str):
            features_to_encode = [features_to_encode]
        strategies = ["keep", "drop"]
        if strategy not in strategies:
            raise ValueError(f"strategy must be one of {strategies}")
        self.strategy = strategy
        self.features_to_encode = features_to_encode

    def fit(
        self,
        x: polars.DataFrame,
        y: Optional[Union[polars.Series, polars.DataFrame]] = None,
    ) -> None:
        """Fit One Hot Encoder.

        Pass as not needed
        """
        return None

    def transform(
        self,
        x: polars.DataFrame,
    ) -> polars.DataFrame:
        """Apply one hot encoding to the provided dataframe.

        Args:
            x (polars.DataFrame): features table to transform

        Returns:
            polars.DataFrame: transformed dataframe
        """
        for feature in self.features_to_encode:
            one_hot = x.select(feature).to_dummies()
            if self.strategy == "drop":
                x = polars.concat([x, one_hot], how="horizontal").drop(feature)
            elif self.strategy == "keep":
                x = polars.concat([x, one_hot], how="horizontal")
        return x