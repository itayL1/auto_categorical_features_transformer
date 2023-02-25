from typing import NoReturn

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def plot_combinations_eval_metric_graphs(
        evaluated_combinations_of_columns_to_transform, metric_to_plot: str
) -> NoReturn:
    combinations_metric_values = [
        comb_res[metric_to_plot] for comb_res in evaluated_combinations_of_columns_to_transform
    ]
    baseline_metric_val = next(
        comb_res[metric_to_plot] for comb_res in evaluated_combinations_of_columns_to_transform
        if comb_res['transformed_columns'] == str([])
    )
    assert baseline_metric_val is not None

    metric_values_series = pd.Series(combinations_metric_values)
    hist_plt = metric_values_series.plot(kind='hist', bins=50, color='orange')
    _color_hist_value_bar(hist_plt, baseline_metric_val, color='blue')

    plt.rcParams['figure.figsize'] = (10, 6)
    plt.title(f'Transformations {metric_to_plot} values')
    plt.legend(handles=[
        Patch(color='blue', label='Baseline value'),
        Patch(color='orange', label='Other combination values')]
    )
    plt.show()
    plt.close()


def _color_hist_value_bar(hist_plt, bar_value_to_color, color) -> NoReturn:
    min_distance = float('inf')
    index_of_bar_to_label = 0
    for i, rectangle in enumerate(hist_plt.patches):
        tmp = abs((rectangle.get_x() + (rectangle.get_width() * (1 / 2))) - bar_value_to_color)
        if tmp < min_distance:
            min_distance = tmp
            index_of_bar_to_label = i
    hist_plt.patches[index_of_bar_to_label].set_color(color)
