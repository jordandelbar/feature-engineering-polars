"""Target encoding.

Target encoding is a technique used in data science to encode categorical variables.

It replaces each categorical value with the mean of the target variable for that value.
"""
import logging
from typing import Any, Dict, List, Union

import polars


class TargetEncoder:
    """Target Encoder class."""

    def __init__(self, smoothing: int, features_to_encode: Union[str, List]):
        """Init.

        Args:
            smoothing (int): smoothing to apply
            features_to_encode (str | list): list of features to encode
        """
        if isinstance(features_to_encode, str):
            features_to_encode = [features_to_encode]
        self.smoothing = smoothing
        self.features_to_encode = features_to_encode
        self.global_mean: Union[int, float, None] = None
        self.mapping: Dict[str, Dict[str, Any]] = dict()

    def _check_features_unique_values(self, x: polars.DataFrame) -> None:
        """Check if the features to impute are numerical.

        Args:
            x (polars.DataFrame): feature dataset

        Returns:
            None
        """
        for feature in self.features_to_encode:
            pct_unique = x[feature].n_unique() / x.height
            if pct_unique >= 0.5 and x[feature].dtype.is_numeric():
                logger = logging.getLogger(__name__)
                logger.warning(f"Feature ['{feature}'] is possibly numerical")

    def fit(
        self, x: polars.DataFrame, y: Union[polars.Series, polars.DataFrame]
    ) -> "TargetEncoder":
        """Fit the target encoder.

        Args:
            x (polars.DataFrame): features table
            y (y: Union[polars.Series, polars.DataFrame]): target

        Returns:
            self
        """
        # Check if the features to impute are numerical and warn the user if not
        self._check_features_unique_values(x)

        if isinstance(y, polars.DataFrame):
            on = y.columns[0]
        else:
            on = y.name

        x = x.with_columns(y)

        # Compute the global mean
        mean = x[on].mean()
        self.global_mean = mean

        for feature in self.features_to_encode:
            # Compute the count and mean of each group
            agg = x.group_by(feature).agg(
                [
                    polars.len().alias("count").cast(polars.Float64),
                    polars.col(on).mean().cast(polars.Float64).alias("mean"),
                ]
            )
            # Compute the smoothed mean
            smooth = agg.with_columns(
                encoding=(
                    polars.col("count") * polars.col("mean")
                    + self.smoothing * mean  # type: ignore
                )
                / (polars.col("count") + self.smoothing)
            ).select([polars.col(feature), polars.col("encoding")])
            self.mapping[feature] = {
                "table": smooth.to_dict(as_series=False),
                "dtype": x.get_column(feature).dtype,
            }
        return self

    def transform(self, x: polars.DataFrame) -> polars.DataFrame:
        """Apply the mapping to the provided dataframe.

        Args:
            x (polars.DataFrame): features table to transform

        Returns:
            polars.DataFrame: transformed dataframe
        """
        features_with_unseen = list()
        for feature in self.mapping.keys():
            # Cast the mapping table
            mapping_table = polars.from_dict(
                self.mapping[feature]["table"]
            ).with_columns(polars.col(feature).cast(self.mapping[feature]["dtype"]))

            # Enforce mapping dtype if different
            if x[feature].dtype != self.mapping[feature]["dtype"]:
                logger = logging.getLogger(__name__)
                logger.warning(
                    msg=(
                        f"Feature ['{feature}'] was mapped "
                        f"with dtype {self.mapping[feature]['dtype']} "
                        f"not {x[feature].dtype}, "
                        f"{self.mapping[feature]['dtype']} was enforced"
                    )
                )
                x = x.with_columns(
                    polars.col(feature).cast(self.mapping[feature]["dtype"])
                )

            temp = x.join(mapping_table, on=feature, how="left")
            x = x.with_columns(temp["encoding"].alias(feature)).select(x.columns)

            # Handling of unseen data
            # TODO: let user choose strategy
            if x[feature].is_null().any():
                features_with_unseen.append(feature)
                x = x.with_columns(
                    polars.col(feature).fill_null(self.global_mean).alias(feature)
                )
        if features_with_unseen:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"{features_with_unseen} have unseen values, defaults to global mean"  # noqa: E501
            )
        return x

    def fit_transform(
        self, x: polars.DataFrame, y: Union[polars.Series, polars.DataFrame]
    ) -> polars.DataFrame:
        """Fit and apply the mapping to the provided dataframe.

        Args:
            x (polars.DataFrame): features table to fit and transform
            y (y: Union[polars.Series, polars.DataFrame]): target

        Returns:
            polars.DataFrame: transformed dataframe
        """
        self.fit(x=x, y=y)
        return self.transform(x=x)
