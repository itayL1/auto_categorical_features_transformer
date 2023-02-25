import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Collection, Optional, Type, Union, NoReturn, Tuple, Dict

import numpy as np
import pandas as pd
import datapane as dp
from category_encoders import OneHotEncoder, WOEEncoder, LeaveOneOutEncoder
from pandas.core.dtypes.common import is_string_dtype
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from auto_categorical_features_transformer.categorical_transformation_methods import CategoricalTransformationMethods
from auto_categorical_features_transformer.common.utils import get_all_subgroups, kfold_dataset_split, \
    is_multiclass_label

from auto_categorical_features_transformer.common.plot_utils import plot_combinations_eval_metric_graphs
from auto_categorical_features_transformer.summary_table_builder import build_summary_table


@dataclass
class CategoricalTransformationResults:
    transformation_candidate_features: Tuple[str]
    baseline_evaluation_results: Dict[str, float]
    summary_table: Union[pd.DataFrame, dp.DataTable]


class CategoricalFeatureTransformationsEvaluator:
    RARE_FEATURE_VALUE = -1_000_000

    def __init__(
            self,
            transformation_method: CategoricalTransformationMethods,
            classifier_cls: Type[BaseEstimator],
            classifier_params: Optional[dict],
            requested_evaluation_metrics: Tuple[str],
            absolute_low_distinct_values_count_threshold: int,
            relative_low_distinct_values_ratio_threshold: float,
            single_feature_max_number_of_new_features: int,
            verbose: bool
    ):
        self._transformation_method = transformation_method
        self._classifier_cls = classifier_cls
        self._classifier_params = classifier_params
        self._requested_evaluation_metrics = requested_evaluation_metrics
        self._absolute_low_distinct_values_count_threshold = absolute_low_distinct_values_count_threshold
        self._relative_low_distinct_values_ratio_threshold = relative_low_distinct_values_ratio_threshold
        self._single_feature_max_number_of_new_features = single_feature_max_number_of_new_features
        self._verbose = verbose

    def create_transformations_report(self, X: pd.DataFrame, y: pd.Series, return_rich_table: bool = True) \
            -> Optional[CategoricalTransformationResults]:
        candidate_columns_for_transformations = self._find_candidate_columns_for_transformations(X)
        if any(candidate_columns_for_transformations):
            logging.info(f'auto_categorical_features_transformer - detected candidate features for categorical '
                         f'transformation: {candidate_columns_for_transformations}')
        else:
            logging.info('auto_categorical_features_transformer - no candidate feasters were found, doing nothing')
            return None

        evaluated_combinations_of_columns_to_transform = self._evaluate_transformation_combinations(
            X, y, candidate_columns_for_transformations
        )
        if self._verbose:
            metric_to_plot = self._requested_evaluation_metrics[0]
            plot_combinations_eval_metric_graphs(evaluated_combinations_of_columns_to_transform, metric_to_plot)

        results = self._build_results_object(
            candidate_columns_for_transformations,
            evaluated_combinations_of_columns_to_transform,
            return_rich_table
        )
        return results

    def _build_results_object(
            self,
            candidate_columns_for_transformations,
            evaluated_combinations_of_columns_to_transform,
            return_rich_table: bool
    ) -> CategoricalTransformationResults:
        evaluated_combinations_of_columns_to_transform_df = pd.DataFrame(
            data=evaluated_combinations_of_columns_to_transform
        )

        baseline_evaluation_results = self._extract_baseline_results(
            evaluated_combinations_of_columns_to_transform_df
        )
        summary_table = build_summary_table(
            evaluated_combinations_of_columns_to_transform_df, baseline_evaluation_results,
            self._requested_evaluation_metrics, self._verbose
        )
        if return_rich_table:
            summary_table = dp.DataTable(summary_table)
        results = CategoricalTransformationResults(
            transformation_candidate_features=tuple(candidate_columns_for_transformations),
            baseline_evaluation_results=baseline_evaluation_results,
            summary_table=summary_table
        )
        return results

    @staticmethod
    def _extract_baseline_results(evaluated_combinations_of_columns_to_transform_df):
        baseline_results = evaluated_combinations_of_columns_to_transform_df[
            evaluated_combinations_of_columns_to_transform_df['transformed_columns'] == str([])
        ]
        assert baseline_results.shape[0] == 1, 'expected to find 1 row exactly'
        baseline_evaluation_results = baseline_results.iloc[0]
        return baseline_evaluation_results

    def _evaluate_transformation_combinations(self, X, y, candidate_columns_for_transformations):
        combinations_of_columns_to_transform = get_all_subgroups(candidate_columns_for_transformations)
        evaluated_combinations_of_columns_to_transform = []
        for columns_to_transform in tqdm(combinations_of_columns_to_transform, desc='transformations evaluation'):
            test_evaluation_results = self._train_and_evaluate_classifier(X, y, columns_to_transform)
            evaluated_combinations_of_columns_to_transform.append({
                'transformed_columns': str(list(columns_to_transform)),
                **test_evaluation_results
            })
        return evaluated_combinations_of_columns_to_transform

    def _train_and_evaluate_classifier(self, X: pd.DataFrame, y: pd.Series, columns_to_transform_comb: Collection[str]):
        scores_average_strategy = 'weighted' if is_multiclass_label(y) else 'binary'
        supported_eval_metric_to_func = {
            'accuracy': accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average=scores_average_strategy),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average=scores_average_strategy),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average=scores_average_strategy),
        }
        self._validate_user_defined_eval_metrics_are_supported(supported_eval_metric_to_func.keys())

        eval_metric_to_fold_test_values = defaultdict(list)
        for X_train, X_test, y_train, y_test in kfold_dataset_split(X, y):
            X_train_transformed_df, X_test_transformed_df = self._apply_categorical_transformation_on_features(
                X_train, X_test, y_train, columns_to_transform_comb)
            X_train_final_df, X_test_final_df = self._dataset_basic_preprocessing(
                X_train_transformed_df, X_test_transformed_df)

            clf = self._classifier_cls(**(self._classifier_params or {}))
            clf.fit(X_train_final_df, y_train)
            y_test_pred = clf.predict(X_test_final_df)

            for metric_name in self._requested_evaluation_metrics:
                metric_func = supported_eval_metric_to_func[metric_name]
                metric_val = metric_func(y_test, y_test_pred)
                eval_metric_to_fold_test_values[metric_name].append(metric_val)

        test_evaluation_results = {}
        for metric_name, folds_test_values in eval_metric_to_fold_test_values.items():
            test_evaluation_results[metric_name] = np.mean(folds_test_values)
            test_evaluation_results[f'{metric_name}_std'] = np.std(folds_test_values)

        return test_evaluation_results

    def _validate_user_defined_eval_metrics_are_supported(self, supported_eval_metrics: Collection[str]) -> NoReturn:
        not_supported_user_defined_eval_metrics = [
            metric_name not in supported_eval_metrics for metric_name in self._requested_evaluation_metrics
        ]
        assert not any(not_supported_user_defined_eval_metrics), \
            f'the following evaluation metrics are not supported yet - {not_supported_user_defined_eval_metrics}. ' \
            f'the metrics that are currently supported are: {list(supported_eval_metrics)}'

    def _apply_categorical_transformation_on_features(
            self, X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, features_to_transform: Collection[str]
    ):
        X_train = X_train.copy()
        X_test = X_test.copy()

        not_candidate_columns = list(set(X_train.columns) - set(features_to_transform))
        X_train_ax1_parts = [X_train[not_candidate_columns]] if any(not_candidate_columns) else []
        X_test_ax1_parts = [X_test[not_candidate_columns]] if any(not_candidate_columns) else []
        for feature_name in features_to_transform:
            x_train_feature = X_train[feature_name]
            x_test_feature = X_test[feature_name]
            x_train_transformed_feature_df, x_test_transformed_feature_df = \
                self._apply_categorical_feature_transformation(
                    x_train_feature, x_test_feature, y_train, self._transformation_method,
                )
            X_train_ax1_parts.append(x_train_transformed_feature_df)
            X_test_ax1_parts.append(x_test_transformed_feature_df)

        X_train_transformed_df = pd.concat(X_train_ax1_parts, axis=1)
        X_test_transformed_df = pd.concat(X_test_ax1_parts, axis=1)
        return X_train_transformed_df, X_test_transformed_df

    def _apply_categorical_feature_transformation(
            self,
            x_train_feature: pd.Series,
            x_test_feature: pd.Series,
            y_train: pd.Series,
            transformation_method: CategoricalTransformationMethods
    ):
        x_train_feature, x_test_feature = self._map_non_frequent_feature_values_to_other(
            x_train_feature, x_test_feature, self._single_feature_max_number_of_new_features,
            other_value=self.RARE_FEATURE_VALUE
        )

        if transformation_method is CategoricalTransformationMethods.OneHot:
            one_hot_encoder = OneHotEncoder()
            x_train_transformed_feature_df = one_hot_encoder.fit_transform(x_train_feature.astype(str))
            x_test_transformed_feature_df = one_hot_encoder.transform(x_test_feature.astype(str))
        elif transformation_method is CategoricalTransformationMethods.WeightOfEvidence:
            woe_encoder = WOEEncoder()
            x_train_transformed_feature_df = woe_encoder.fit_transform(x_train_feature.astype(str), y_train)
            x_test_transformed_feature_df = woe_encoder.transform(x_test_feature.astype(str))
        elif transformation_method is CategoricalTransformationMethods.LeaveOneOut:
            woe_encoder = LeaveOneOutEncoder()
            x_train_transformed_feature_df = woe_encoder.fit_transform(x_train_feature.astype(str), y_train)
            x_test_transformed_feature_df = woe_encoder.transform(x_test_feature.astype(str))
        else:
            raise NotImplementedError(f"unknown transformation_method - '{transformation_method}'")
        return x_train_transformed_feature_df, x_test_transformed_feature_df

    @staticmethod
    def _map_non_frequent_feature_values_to_other(
            x_train_feature: pd.Series, x_test_feature: pd.Series,
            max_number_of_output_features: int, other_value
    ):
        train_feature_values_sorted_by_frequency_desc = list(x_train_feature.value_counts(ascending=False).index)
        train_feature_values_with_max_frequency = train_feature_values_sorted_by_frequency_desc[:(max_number_of_output_features - 1)]

        x_train_adjusted_feature = x_train_feature.map(
            {val: val for val in train_feature_values_with_max_frequency}
        ).fillna(other_value)
        x_test_adjusted_feature = x_test_feature.map(
            {val: val for val in train_feature_values_with_max_frequency}
        ).fillna(other_value)
        return x_train_adjusted_feature, x_test_adjusted_feature

    def _find_candidate_columns_for_transformations(self, X) -> set:
        candidate_columns_for_transformations = {
            col for col in X.columns
            if self._is_candidate_for_categorical(X[col])
        }
        return candidate_columns_for_transformations

    def _dataset_basic_preprocessing(self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
        X_train_df = X_train_df.copy()
        X_test_df = X_test_df.copy()
        X_train_df, X_test_df = self._string_to_numeric_columns(X_train_df, X_test_df)

        scaler = StandardScaler()
        X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df.values), columns=X_train_df.columns, index=X_train_df.index)
        X_test_df = pd.DataFrame(scaler.transform(X_test_df.values), columns=X_test_df.columns, index=X_test_df.index)

        X_train_df.fillna(-1, inplace=True)
        X_test_df.fillna(-1, inplace=True)

        return X_train_df, X_test_df

    @staticmethod
    def _string_to_numeric_columns(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
        X_train_df = X_train_df.copy()
        X_test_df = X_test_df.copy()

        for col in list(X_train_df.columns):
            if is_string_dtype(X_train_df[col]):
                all_possible_col_values = set(X_train_df[col]) | set(X_test_df[col])
                val_to_id = {val: idx for idx, val in enumerate(all_possible_col_values)}
                X_train_df[col] = X_train_df[col].map(val_to_id).astype(float)
                X_test_df[col] = X_test_df[col].map(val_to_id).astype(float)
        return X_train_df, X_test_df

    def _is_candidate_for_categorical(self, col: pd.Series) -> bool:
        number_of_distinct_values = col.nunique()
        has_absolute_low_number_of_distinct_values = \
            number_of_distinct_values <= self._absolute_low_distinct_values_count_threshold

        ratio_of_distinct_values = number_of_distinct_values / len(col)
        has_relative_low_ratio_of_distinct_values = \
            ratio_of_distinct_values <= self._relative_low_distinct_values_ratio_threshold

        return has_absolute_low_number_of_distinct_values or has_relative_low_ratio_of_distinct_values
