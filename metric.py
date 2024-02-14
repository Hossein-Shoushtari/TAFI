import numpy as np


def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    return np.sqrt(np.mean((est - gt) ** 2))

def compute_ate_rte(est, gt,pred_per_min=60):
    ate = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        ratio = pred_per_min / est.shape[0]
        rte = compute_absolute_trajectory_error(est, gt) * ratio
    else:
        rte = compute_absolute_trajectory_error(est, gt)
    return ate, rte

def errors_sqrt(prediction, target):
    AI_errors = np.array(
        [
            np.sqrt(
                (prediction[i, 0] - target[i, 0]) ** 2
                + (prediction[i, 1] - target[i, 1]) ** 2
            )
            for i in range(len(prediction))
        ])
    return AI_errors


def CDF_params(AI_errors):
    # * 50/90 lines
    CEP50_index = np.abs(np.linspace(0, 1, AI_errors.shape[0]) - 0.5).argmin()
    CEP90_index = np.abs(np.linspace(0, 1, AI_errors.shape[0]) - 0.9).argmin()
    CEP50 = np.sort(AI_errors)[CEP50_index]
    CEP90 = np.sort(AI_errors)[CEP90_index]

    return CEP50, CEP90


def distance_time_below_threshold(AI_errors, prediction, threshold):
    errors_below_1m = []
    for j in range(len(AI_errors)):
        if AI_errors[j] < threshold:
            errors_below_1m.append(list(prediction[j]))
        else:
            break
    errors_below_1m = np.array(errors_below_1m)
    # if errors_below_1m is empty, set time and distance to 0
    if len(errors_below_1m) == 0:
        time_below_1m = 0
        distance_below_1m = 0
    else:
        time_below_1m = np.size(errors_below_1m)  # s
        distance_below_1m = sum(
            [
                np.sqrt(
                    (errors_below_1m[:, 0][i - 1] - errors_below_1m[:, 0][i]) ** 2
                    + (errors_below_1m[:, 1][i - 1] - errors_below_1m[:, 1][i]) ** 2
                )
                for i in range(1, len(errors_below_1m[:, 0]))
            ]
        )  # m
    return time_below_1m, distance_below_1m


def compute_heading_error(est, gt):
    """
    Args:
        est: the estimated heading as sin, cos values
        gt: the ground truth heading as sin, cos values
    Returns:
        MSE error and angle difference from dot product
    """

    mse_error = np.mean((est-gt)**2)
    dot_prod = np.sum(est * gt, axis=1)
    angle = np.arccos(np.clip(dot_prod, a_min=-1, a_max=1))

    return mse_error, angle
