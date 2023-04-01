"""Most frequent.

Replacing the null values with the most frequent values of the variable
"""
# TODO: mean / mode / median share the same pattern and should be one class

from typing import Dict, List, Union

import polars


class MostFrequentImputer:
    """Most frequent imputer class."""

    def __init__(self, features_to_impute: Union[str, List]):
        """Init.

        Args:
            features_to_impute (list): list of feature to impute
        """
        self.features_to_impute = features_to_impute
        self.mapping: Dict[str, float] = dict()

    def fit(self, x: polars.DataFrame):
        """Fit.

        Args:
            x (polars.DataFrame): feature dataset

        Returns:
            None
        """
        if isinstance(self.features_to_impute, str):
            self.features_to_impute = [self.features_to_impute]

        for features in self.features_to_impute:
            self.mapping[features] = x[features].mode()
        return None

    def transform(self, x: polars.DataFrame) -> polars.DataFrame:
        """Transform.

        Args:
            x (polars.DataFrame): feature dataset

        Returns:
            polars.DataFrame: transformed dataset
        """
        for feature in self.mapping.keys():
            x = x.with_columns(
                polars.col(feature).fill_null(
                    polars.lit(self.mapping[feature]),
                )
            )
        return x

    def fit_transform(self, x: polars.DataFrame) -> polars.DataFrame:
        """Fit & transform.

        Args:
            x (polars.DataFrame): feature dataset
            features_list (list): list of features to mean impute

        Returns:
            polars.DataFrame: transformed dataset
        """
        self.fit(x=x)
        return self.transform(x=x)
