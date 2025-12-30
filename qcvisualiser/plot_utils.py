import numpy as np
from operator import le, gt

def filter_points(points, x_range, y_range, eps=1e-6):
    if x_range is None or y_range is None:
        return points

    # if np.isnan(x_range + y_range).any():
    #     return points
    # NOTE: This attempts to avoid a weird edge case where all points have the
    # same value. This causes in equal limits on the range which results in
    # no points being selected for plotting.
    if x_range[0] == x_range[1]:
        x_range = (x_range[0] - eps, x_range[1] + eps)
    if y_range[0] == y_range[1]:
        y_range = (y_range[0] - eps, y_range[1] + eps)

    return points[x_range, y_range]

def threshold_points(points, threshold=50000, inverse=False):
    op = le if inverse else gt

    if op(len(points), threshold):
        return points
    else:
        return points.iloc[:0]
