import itertools

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_all_subgroups(group: set):
    all_subsets = []
    for n in range(len(group) + 1):
        all_subsets.extend(
            list(sorted(comb)) for comb in itertools.combinations(group, r=n)
        )
    return all_subsets


def kfold_dataset_split(X: pd.DataFrame, y: pd.Series):
    k_fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_train_index, fold_test_index in k_fold_splitter.split(X, y):
        X_train = X.loc[fold_train_index]
        y_train = y.loc[fold_train_index]
        X_test = X.loc[fold_test_index]
        y_test = y.loc[fold_test_index]
        yield X_train, X_test, y_train, y_test


def is_multiclass_label(y: pd.Series) -> bool:
    return y.nunique() > 2
