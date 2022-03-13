import numpy as np


def calc_abe(x_true, x_pred) -> np.ndarray:
    """Compute Average displacement error (ade).

    In the original paper, ade is mean square error (mse) over all estimated
    points of a trajectory and the true points.

    :param x_true: (n_samples, seq_len, max_n_peds, 3)
    :param x_pred: (n_samples, seq_len, max_n_peds, 3)
    :return: Average displacement error
    """
    return np.mean(np.square(x_true-x_pred))


def calc_fde(x_true, x_pred) -> np.ndarray:
    """Compute Final displacement error (fde).

    In the original paper, ade is mean square error (mse) over all estimated
    points of a trajectory and the true points.

    :param x_true: (n_samples, seq_len, max_n_peds, 3)
    :param x_pred: (n_samples, seq_len, max_n_peds, 3)
    :return: Average displacement error
    """
    pos_final_true = x_true[:, -1, :]
    pos_final_pred = x_pred[:, -1, :]

    return np.mean(np.square(pos_final_true-pos_final_pred))
