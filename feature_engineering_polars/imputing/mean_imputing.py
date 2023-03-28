"""Mean imputing.

Mean imputation is a method of handling missing data by replacing missing values
with the mean value of the entire feature.
"""
from typing import List, Union

import polars


class MeanImputer:
    """Mean imputer class."""

    def __init__(self, features_to_impute: Union[str, List]):
        """Init.

        Args:
            features_to_impute (list): list of feature to impute
        """
        self.features_to_impute = features_to_impute
        self.mapping = dict()

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
            self.mapping[features] = x[features].mean()
        return None

    def transform(self, x: polars.DataFrame):
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

    def fit_transform(self, x: polars.DataFrame):
        """Fit & transform.

        Args:
            x (polars.DataFrame): feature dataset
            features_list (list): list of features to mean impute

        Returns:
            polars.DataFrame: transformed dataset
        """
        self.fit(x=x)
        return self.transform(x=x)
