import numpy as np
from operator import lt, gt

def filter_points(points, x_range, y_range):
    if x_range is None or y_range is None:
        return points
    if np.isnan(x_range + y_range).any():
        return points
    return points[x_range, y_range]

def threshold_points(points, threshold=50000, inverse=False):
    op = lt if inverse else gt

    if op(len(points), threshold):
        return points
    else:
        return points.iloc[:0]
