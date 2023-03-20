'''
File: distances.py
Project: fid-scoring-mri
File Created: 2023-03-20 21:01:17
Author: sangminlee
-----
This script ...
Reference
...
'''
import numpy as np
from scipy.spatial import distance


class EuclideanDistance:
    def __call__(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))


class CosineSimilarity:
    def __call__(self, x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class MahalanobisNorm:
    def __call__(self, x, y):
        return distance.mahalanobis(x, y)
