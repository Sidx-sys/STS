import numpy as np


def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    cos_sim = v1.dot(v2) / (n1 * n2)
    return (cos_sim + 1) / 2
