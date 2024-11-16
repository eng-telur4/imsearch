import numpy as np


def normalize_vector(vector):
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)
