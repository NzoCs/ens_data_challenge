# =============================================================================
# TRANSFORMS - Target transformation functions
# =============================================================================

import numpy as np


def transform_y(y: np.ndarray) -> np.ndarray:
    """Transform y via rank."""
    n = len(y)
    ranks = (-y).argsort().argsort() + 1
    return ranks / (n + 1)


def scale_01(x: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-10)
