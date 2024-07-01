from typing import Union

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)


def generate_experiment_name(reg_coeff: float, lr: float) -> (str, str):
    return f"Reg_{reg_coeff}_LR_{lr}"


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.array(sklearn_confusion_matrix(targets, predictions))
