import numpy as np


def mu_func(v: float) -> float:
    """Models how the velocity (v) is transformed into the coefficient mu."""
    return __baseline(v)  # to change

def __baseline(v: float) -> float:
    """Baseline function for mu_func. DO NOT ALTER THIS"""
    return 1.0 + (1.72 - 1.0) * np.exp(-((np.abs(v) / 3.7) ** 1.37))

