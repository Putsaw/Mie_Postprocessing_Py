import numpy as np


def calc_circle(p1, p2, p3):
    """Return center and radius of the circle defined by three points.

    Parameters
    ----------
    p1, p2, p3 : tuple of int or float
        Three points as ``(x, y)`` pairs.

    Returns
    -------
    center : tuple of float
        Calculated circle center ``(cx, cy)``.
    radius : float
        Circle radius.
    """

    A = np.array([
        [p1[0], p1[1], 1],
        [p2[0], p2[1], 1],
        [p3[0], p3[1], 1],
    ], dtype=float)
    B = np.array([
        p1[0] ** 2 + p1[1] ** 2,
        p2[0] ** 2 + p2[1] ** 2,
        p3[0] ** 2 + p3[1] ** 2,
    ], dtype=float)

    coeff = np.linalg.solve(A, B)
    cx = 0.5 * coeff[0]
    cy = 0.5 * coeff[1]
    r = np.sqrt(cx ** 2 + cy ** 2 + coeff[2])
    return (cx, cy), r