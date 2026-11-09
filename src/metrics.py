import numpy as np


def accuracy(y_true, y_pred):
    """Share of rows where the prediction equals the true label."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, n_classes=5):
    """C[j, k] = number of rows with true class j predicted as class k."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def per_class_accuracy(C):
    """For class j: C[j, j] / sum_k C[j, k] (share of class-j beats recognised)."""
    return np.diag(C) / C.sum(axis=1)


def balanced_accuracy(C):
    """Average of the per-class accuracies; every class counts equally."""
    return float(per_class_accuracy(C).mean())
