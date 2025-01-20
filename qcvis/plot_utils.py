import numpy as np
from operator import lt, gt

def filter_points(points, x_range, y_range, eps=1e-6):
    if x_range is None or y_range is None:
        print("A", x_range, y_range)
        return points
    if np.isnan(x_range + y_range).any():
        print("B", x_range, y_range)
        return points

    if y_range[0] == y_range[1]:
        y_range = (y_range[0] - eps, y_range[1] + eps)

    return points[x_range, y_range]

def threshold_points(points, threshold=50000, inverse=False):
    op = lt if inverse else gt
    print(op, len(points), threshold)

    if op(len(points), threshold):
        return points
    else:
        return points.iloc[:0]
