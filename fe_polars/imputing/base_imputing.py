"""Imputing.

Imputation is a method of handling missing data by replacing missing values
with another value. You can choose between different strategies:

- Mean imputing: replace with the mean of the non-null records
- Median imputing: replace with the median of the non-null records.
- Mode imputing: replace with the most frequent records of the variable.
- Max imputing: replace with the maximum value of the records.
- Min imputing: replace with the minimum value of the records.
"""
import logging
from typing import Dict, List, Optional, Union

import polars


class Imputer:
    """Imputer class.

    Impute a value in place of the Null records in the dataframe
    depending on the strategy you choose.

    Available strategies:

    - Mean: strategy="mean"
    - Median: strategy="median"
    - Mode: strategy="mode"
    - Maximum: strategy="max"
    - Minimum: strategy="min"
    """

    def __init__(self, features_to_impute: Union[str, List], strategy: str = "mean"):
        """Init.

        Args:
            features_to_impute (list): list of feature to impute
            strategy (str): imputing strategy to apply
        """
        strategies = ["mean", "median", "mode", "max", "min"]
        if strategy not in strategies:
            raise ValueError(f"strategy must be one of {strategies}")
        self.features_to_impute = features_to_impute
        self.strategy = strategy
        self.mapping: Dict[str, float] = dict()

    def fit(
        self,
        x: polars.DataFrame,
        y: Optional[Union[polars.DataFrame, polars.Series]] = None,
    ) -> None:
        """Fit.

        Args:
            x (polars.DataFrame): feature dataset
            y (y: Union[polars.Series, polars.DataFrame]): target (not used)

        Returns:
            None
        """
        if isinstance(self.features_to_impute, str):
            self.features_to_impute = [self.features_to_impute]

        # TODO: give the option to define a strategy by feature
        for feature in self.features_to_impute:
            if self.strategy == "mean":
                self.mapping[feature] = x[feature].mean()
            elif self.strategy == "median":
                self.mapping[feature] = x[feature].median()
            elif self.strategy == "mode":
                # If the most frequent value is not null
                if x[feature].mode().item():
                    self.mapping[feature] = x[feature].mode()

                # Else we take the second most frequent value
                else:
                    logger = logging.getLogger(__name__)
                    # Warning: We are sorting the value counts by the feature value
                    # in case there is ex-aequo.
                    self.mapping[feature] = (
                        x[feature]
                        .value_counts()
                        .sort(by=["counts", feature], descending=True)[2, :]
                        .select(feature)
                        .item()
                    )
                    logger.warning(
                        f"Most frequent value for ['{feature}'] "
                        "is Null, defaults to the second most "
                        f"frequent value: {self.mapping[feature]}."
                    )
            elif self.strategy == "max":
                self.mapping[feature] = x[feature].max()
            elif self.strategy == "min":
                self.mapping[feature] = x[feature].min()
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
