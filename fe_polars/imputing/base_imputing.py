"""Imputing.

Imputation is a method of handling missing data by replacing missing values
with another value. You can choose between different strategies:

- Mean imputing: replace with the mean of the non-null records
- Median imputing: replace with the median of the non-null records.
- Mode imputing: replace with the most frequent records of the variable.
- Max imputing: replace with the maximum value of the records.
- Min imputing: replace with the minimum value of the records.
"""
from typing import Optional, Union

import polars


class Imputer:
    """Imputer class.

    Impute a value in place of the Null records in the dataframe
    depending on the strategy you choose.

    Available strategies:

    - Mean: strategy="mean"
    - Median: strategy="median"
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
        valid_params = {
            "features_to_impute",
            "strategy",
            "strategy_dict",
            "fixed_value",
        }
        valid_strategies = {"mean", "median", "max", "min", "fixed_value"}

        # Check that all provided parameters are valid
        for param_name in kwargs:
            if param_name not in valid_params:
                raise ValueError(f"Invalid parameter '{param_name}' provided.")

        self.features_to_impute = kwargs.get("features_to_impute", None)
        self.strategy = kwargs.get("strategy", None)
        self.strategy_dict = kwargs.get("strategy_dict", dict())
        self.fixed_value = kwargs.get("fixed_value", None)
        self.mapping = dict()

        if self.strategy:
            if self.strategy not in valid_strategies:
                raise ValueError(f"strategy must be one of {valid_strategies}")

        if isinstance(self.features_to_impute, str):
            self.features_to_impute = [self.features_to_impute]

        for i in self.strategy_dict.keys():
            if isinstance(self.strategy_dict[i], str):
                self.strategy_dict[i] = [self.strategy_dict[i]]

        if not self.strategy_dict:
            self._map_strategy_dict()

    def _map_strategy_dict(self):
        """Map the strategies for each column."""
        if self.features_to_impute is not None and self.strategy is not None:
            self.strategy_dict = {self.strategy: self.features_to_impute}

        elif self.features_to_impute is not None and self.strategy is None:
            self.strategy_dict = {"mean": self.features_to_impute}

    def _process_strategy(self, strategy, feature, x):
        """Process the different strategy by feature."""
        STRATEGY_FUNCTIONS = {
            "mean": getattr(polars.DataFrame, "mean"),
            "median": getattr(polars.DataFrame, "median"),
            "max": getattr(polars.DataFrame, "max"),
            "min": getattr(polars.DataFrame, "min"),
        }
        # Apply the corresponding function based on the strategy
        if isinstance(x.select(feature), polars.DataFrame):
            self.mapping[feature] = STRATEGY_FUNCTIONS[strategy](
                x.select(feature)
            ).item()

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
            self._map_strategy_dict()

        for strategy in self.strategy_dict.keys():
            for feature in self.strategy_dict[strategy]:
                if not x[feature].is_numeric():
                    raise ValueError(f"{feature} is not a numerical feature")
                self._process_strategy(strategy, feature, x)

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
