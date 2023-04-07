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
from typing import Optional, Union

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

    def __init__(self, **kwargs):
        """Init.

        Args:
            kwargs (dict):
                A dictionnary of optional arguments. Valid arguments include:
                - features_to_impute
                - strategy
                - strategy_dict
        """
        valid_params = {"features_to_impute", "strategy", "strategy_dict"}
        valid_strategies = {"mean", "median", "mode", "max", "min"}

        # Check that all provided parameters are valid
        for param_name in kwargs:
            if param_name not in valid_params:
                raise ValueError(f"Invalid parameter '{param_name}' provided.")

        self.features_to_impute = kwargs.get("features_to_impute", None)
        self.strategy = kwargs.get("strategy", None)
        self.strategy_dict = kwargs.get("strategy_dict", dict())
        self.mapping = dict()

        if self.strategy:
            if self.strategy not in valid_strategies:
                raise ValueError(f"strategy must be one of {valid_strategies}")

        if isinstance(self.features_to_impute, str):
            self.features_to_impute = [self.features_to_impute]

        if not self.strategy_dict:
            self.map_strategy_dict()

    def map_strategy_dict(self):
        """Map the strategies for each column."""
        if self.features_to_impute is not None and self.strategy is not None:
            self.strategy_dict = {
                feature: self.strategy for feature in self.features_to_impute
            }

        elif self.features_to_impute is not None and self.strategy is None:
            self.strategy_dict = {
                feature: "mean" for feature in self.features_to_impute
            }

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
        # If no strategy dictionnary has been provided and neither was a list of feature
        # to impute, then we apply the strategy on all the columns that contain
        # null values
        if not self.strategy_dict and self.features_to_impute is None:
            self.features_to_impute = [
                col
                for col in x.columns
                if x[col].is_null().any() and x[col].is_numeric()
            ]
            self.map_strategy_dict()

        for feature in self.strategy_dict.keys():
            if self.strategy_dict[feature] == "mean":
                self.mapping[feature] = x[feature].mean()

            elif self.strategy_dict[feature] == "median":
                self.mapping[feature] = x[feature].median()

            elif self.strategy_dict[feature] == "mode":
                # FIXME: This part should be discuted at some point to
                # agree upon a smart way to handle this based on use cases.
                if x[feature].dtype in [
                    polars.Float32,
                    polars.Float64,
                    polars.Boolean,
                    polars.Decimal,
                ]:
                    raise TypeError(
                        f"dtype `{x[feature].dtype}` is not supported for mode strategy"
                    )
                # If there is only one value that is the most-frequent
                if x[feature].mode().shape[0] <= 1:
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
                            .sort(by=["counts", feature], descending=True)[1, :]
                            .select(feature)
                            .item()
                        )
                        logger.warning(
                            f"Most frequent value for ['{feature}'] "
                            "is Null, defaults to the second most "
                            f"frequent value: {self.mapping[feature]}."
                        )
                else:
                    # We take the first most frequent value
                    self.mapping[feature] = x[feature].mode()[0]

            elif self.strategy_dict[feature] == "max":
                self.mapping[feature] = x[feature].max()
            elif self.strategy_dict[feature] == "min":
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
