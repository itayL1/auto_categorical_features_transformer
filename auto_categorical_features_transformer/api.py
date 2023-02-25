from types import MappingProxyType as ImmutableDict
from typing import Type, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from auto_categorical_features_transformer.categorical_feature_transformations_evaluator import \
    CategoricalFeatureTransformationsEvaluator, CategoricalTransformationResults
from auto_categorical_features_transformer.categorical_transformation_methods import CategoricalTransformationMethods


def create_categorical_feature_transformations_report(
        X: pd.DataFrame,
        y: pd.Series,
        transformation_method: CategoricalTransformationMethods,
        classifier_cls: Type[BaseEstimator] = LogisticRegression,
        classifier_params: Optional[dict] = ImmutableDict(dict(class_weight='balanced')),
        evaluation_metrics: Tuple[str] = ('f1', 'accuracy'),
        absolute_low_distinct_values_count_threshold: int = 10,
        relative_low_distinct_values_ratio_threshold: float = 0.01,  # < 1%
        single_feature_max_number_of_new_features: int = 20,
        return_rich_table: bool = True,
        verbose: bool = False
) -> Optional[CategoricalTransformationResults]:
    """
    Find candidates features for categorical features transformations, and evaluate their combinations
    using the classifier and evaluation metrics provided. If no candidate features were found, do nothing.
    :param X: The features of the dataset
    :param y: The prediction target feature of the classification task at hand
    :param transformation_method: The kind of categorical transformation to use (see the
    CategoricalTransformationMethods enum for the currently supported methods)
    :param classifier_cls: The class of the classifier to use in order to evaluate each transformations combination
    :param classifier_params: Optional parameters for the classifier_cls
    :param evaluation_metrics: a collection of classification evaluation metrics to calculate for each combination.
    The supported metrics are 'accuracy', 'f1', 'precision' nad 'recall'.
    :param absolute_low_distinct_values_count_threshold: The upper bound of number of distinct values in a feature
    that is considered low, and will mark it as a candidate for categorical transformation.
    :param relative_low_distinct_values_ratio_threshold: The upper bound of the ratio between the distinct number
    of values in a feature and the number of rows in the dataset that is considered low, and will mark it as
     a candidate for categorical transformation.
    :param single_feature_max_number_of_new_features: The maximume number of new features that will replace a
    feature after the categorical transformation.
    :param return_rich_table: If set to True, the summary table that will be return in the
    CategoricalTransformationResults object will be datapane's DateTable instead of regular dataframe. This option
    in useful when the summary table is being displayed interactively from a jupyter notebook.
    :param verbose: When toggled on, additional statistics and plots are included during the execution of the function.
    :return:
    """

    transformations_evaluator = CategoricalFeatureTransformationsEvaluator(
        transformation_method,
        classifier_cls,
        classifier_params,
        evaluation_metrics,
        absolute_low_distinct_values_count_threshold,
        relative_low_distinct_values_ratio_threshold,
        single_feature_max_number_of_new_features,
        verbose
    )
    output_report = transformations_evaluator.create_transformations_report(X, y, return_rich_table)
    return output_report
