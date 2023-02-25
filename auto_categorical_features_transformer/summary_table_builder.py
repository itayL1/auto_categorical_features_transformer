from typing import NoReturn

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def build_summary_table(
        evaluated_combinations_of_columns_to_transform_df,
        baseline_evaluation_results,
        evaluation_metrics,
        verbose
) -> pd.DataFrame:
    ordered_display_columns = ['transformed_columns']
    for metric_name in evaluation_metrics:
        evaluated_combinations_of_columns_to_transform_df[
            f'{metric_name}_diff'] = _build_metric_baseline_diff_display_column(
            evaluated_combinations_of_columns_to_transform_df, metric_name, baseline_evaluation_results)

        ordered_display_columns.extend((metric_name, f'{metric_name}_diff'))
        if verbose:
            ordered_display_columns.append(f'{metric_name}_std')

    summary_table_df = evaluated_combinations_of_columns_to_transform_df\
        [ordered_display_columns].sort_values(by=evaluation_metrics[0], ascending=False)

    _round_numeric_columns(summary_table_df, decimals=3)
    return summary_table_df


def _build_metric_baseline_diff_display_column(
        evaluated_combinations_of_columns_to_transform_df, metric_name, baseline_results_row
):
    def _to_diff_display_str(curr_row: pd.Series) -> str:
        baseline_metric_value = baseline_results_row[metric_name]
        curr_row_metric_value = curr_row[metric_name]

        diff = curr_row_metric_value - baseline_metric_value
        diff_rounded = round(diff, 3)
        diff_percentage = (diff / baseline_metric_value) * 100
        diff_percentage_rounded = round(diff_percentage, 1)

        if diff_rounded > 0:
            return f"(↑) {diff_rounded}/ [{diff_percentage_rounded}%]"
        elif diff_rounded == 0:
            return "(-) 0 [0%]"
        else:
            return f"(↓) {diff_rounded} [{diff_percentage_rounded}%]"

    diff_display_column = evaluated_combinations_of_columns_to_transform_df.apply(_to_diff_display_str, axis=1)
    return diff_display_column


def _round_numeric_columns(summary_table_df: pd.DataFrame, decimals: int) -> NoReturn:
    for col in list(summary_table_df.columns):
        if is_numeric_dtype(summary_table_df[col]):
            summary_table_df[col] = summary_table_df[col].round(decimals)
