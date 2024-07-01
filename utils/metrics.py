import numpy as np


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            correct += 1

    return correct / len(predictions)

def accuracy_mat(conf_matrix: np.ndarray) -> float:
    return np.trace(conf_matrix) / np.sum(conf_matrix)

def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    k = np.max(targets) + 1
    conf_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            conf_matrix[i][j] = np.count_nonzero((predictions == j) & (targets == i))
    return conf_matrix
