
import numpy as np



def parameterized_sigmoid(x, x1, x2):
    a = 3 / (x2 - x1)        # scaling
    b = -1.5 - a * x1        # translation
    return 1 / (1 + np.exp(-(a * x + b)))


def piecewise_linear(x, x1, x2):
    y = np.zeros_like(x)
    slope = 1 / (x2 - x1)
    # Linear part between x1 and x2
    mask = (x >= x1) & (x <= x2)
    y[mask] = slope * (x[mask] - x1)
    # Cap at 1 for values above x2
    y[x > x2] = 1
    return y