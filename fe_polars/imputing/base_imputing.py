"""Imputing.

Imputation is a method of handling missing data by replacing missing values
with another value. You can choose between different strategies:

- Mean imputing: replace with the mean of the non-null records
- Median imputing: replace with the median of the non-null records.
- Max imputing: replace with the maximum value of the records.
- Min imputing: replace with the minimum value of the records.
- Fixed value imputing: replace with an arbitrary number.
"""
from typing import Optional, Union

import polars


class Imputer:
    """Imputer class.

    Impute a value in place of the null records in the dataframe
    depending on the strategy chosen.

    Args:
        feature_to_impute (list): list of features to impute
        strategy (str): imputation strategy
        fixed_value (float): specific value to be imputed
                             (with "fixed_value" strategy)
        strategy_dict (dict): dictionnary describing strategies to apply
                              by feature
    """

    def __init__(self, **kwargs):
        """Init.

        Args:
            kwargs (dict):
                A dictionnary of optional arguments. Valid arguments include:
                - features_to_impute
                - strategy
                - strategy_dict
                - fixed_value
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

        # Check if kwargs are compatible together
        if "strategy_dict" in kwargs and (
            "strategy" in kwargs
            or "features_to_impute" in kwargs
            or "fixed_value" in kwargs
        ):
            raise ValueError(
                "Cannot use 'strategy_dict' with 'features_to_impute'"
                " or 'strategy' or 'fixed_value'"
            )

        self.features_to_impute = kwargs.get("features_to_impute", None)
        self.strategy = kwargs.get("strategy", None)
        self.strategy_dict = kwargs.get("strategy_dict", dict())
        self._fit_strategy_dict = False
        self.fixed_value = kwargs.get("fixed_value", None)
        self.mapping = dict()

        if self.strategy:
            if self.strategy not in valid_strategies:
                raise ValueError(f"strategy must be one of {valid_strategies}")

        if isinstance(self.features_to_impute, str):
            self.features_to_impute = [self.features_to_impute]

        if self.strategy_dict:
            for i in self.strategy_dict.keys():
                if i not in valid_strategies:
                    raise ValueError(f"strategy must be one of {valid_strategies}")
                if isinstance(self.strategy_dict[i], str):
                    self.strategy_dict[i] = [self.strategy_dict[i]]
        else:
            self._map_strategy_dict()

    def _check_strategy(self):
        """Check strategy to map the strategy_dict correctly."""
        if self.strategy is None and self.fixed_value is None:
            _strategy = "mean"

        elif self.strategy is not None and self.fixed_value is None:
            _strategy = self.strategy

        elif self.strategy is None and self.fixed_value is not None:
            _strategy = "fixed_value"

        else:
            _strategy = self.strategy
        return _strategy

    def _map_strategy_dict(self):
        """Map the strategies for each column."""
        if self.features_to_impute is None:
            self._fit_strategy_dict = True
            _feature_to_impute = list()
            _strategy = self._check_strategy()

        else:
            _feature_to_impute = self.features_to_impute
            _strategy = self._check_strategy()
            if _strategy == "fixed_value":
                _feature_to_impute = {
                    i: self.fixed_value for i in self.features_to_impute
                }

        self.strategy_dict = {_strategy: _feature_to_impute}

    def _process_strategy(self, strategy, feature, x):
        """Process the different strategy by feature."""
        STRATEGY_FUNCTIONS = {
            "mean": getattr(polars.DataFrame, "mean"),
            "median": getattr(polars.DataFrame, "median"),
            "max": getattr(polars.DataFrame, "max"),
            "min": getattr(polars.DataFrame, "min"),
        }
        # Apply the corresponding function based on the strategy
        if strategy != "fixed_value":
            if isinstance(x.select(feature), polars.DataFrame):
                self.mapping[feature] = STRATEGY_FUNCTIONS[strategy](
                    x.select(feature)
                ).item()
        else:
            self.mapping[feature] = self.strategy_dict["fixed_value"][feature]

    def fit(
        self,
        x: polars.DataFrame,
        y: Optional[Union[polars.DataFrame, polars.Series]] = None,
    ) -> "Imputer":
        """Fit.

        Args:
            x (polars.DataFrame): feature dataset
            y (y: Union[polars.Series, polars.DataFrame]): target (not used)

        Returns:
            self
        """
        # If no strategy dictionnary has been provided and neither was a list of feature
        # to impute, then we apply the strategy on all the columns that contain
        # null values
        if self._fit_strategy_dict:
            self.features_to_impute = [
                col
                for col in x.columns
                if x[col].is_null().any() and x[col].dtype.is_numeric()
            ]
            self._map_strategy_dict()

        for strategy in self.strategy_dict.keys():
            for feature in self.strategy_dict[strategy]:
                if not x[feature].dtype.is_numeric():
                    raise ValueError(f"{feature} is not a numerical feature")
                self._process_strategy(strategy, feature, x)

        return self

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
