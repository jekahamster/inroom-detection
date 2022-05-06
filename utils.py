import os
import pathlib
import numpy as np
import cv2


EPS = 1e-9


def imread(path):
    assert pathlib.Path(path).exists(), f"File {path} does not exists"
    return cv2.imread(str(path))


def proportional_resize(image, max_size=900):
    h, w, *_ = image.shape
    coeff = max_size / max(h, w)
    return cv2.resize(src=image, dsize=(int(w*coeff), int(h*coeff)))


def to_homo_coords(coords):
    return np.array([
        [point[0], point[1], 1.]
        for point in coords
    ], dtype=np.float32)


def from_homo_coords(coords):
    return np.array([
        [point[0] / point[2], point[1] / point[2]]
        for point in coords
    ])


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return np.fabs(self.x - other.x) < EPS and np.fabs(self.y - other.y) < EPS
