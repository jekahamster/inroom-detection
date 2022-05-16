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


def sort_4_points(points):
    points_arr = np.array(points[:4], dtype=np.float32).reshape(4, 2)
    sorted_by_y = points_arr[points_arr[:, 1].argsort()]
    
    _temp = sorted_by_y[:2]
    _temp = _temp[_temp[:, 0].argsort()]
    top_left = _temp[0]
    top_right = _temp[1]

    _temp = sorted_by_y[2:]
    _temp = _temp[_temp[:, 0].argsort()]
    bottom_left = _temp[0]
    bottom_right = _temp[1]

    return top_left, top_right, bottom_left, bottom_right


def rect_sizes(top_left, top_right, bottom_left, bottom_right):
    width_a = np.sqrt((top_left[0] - top_right[0])**2 + (top_left[1] - top_right[1])**2 )
    width_b = np.sqrt((bottom_left[0] - bottom_right[0])**2 + (bottom_left[1] - bottom_right[1])**2 )
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt((top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2 )
    height_b = np.sqrt((top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2 )
    max_height = max(int(height_a), int(height_b))
    return max_width, max_height, width_a, width_b, height_a, height_b


def get_perspective_matrix(points):
    points = points[:4]
    top_left, top_right, bottom_left, bottom_right = sort_4_points(points)
    max_width, max_height, *_ = rect_sizes(top_left, top_right, bottom_left, bottom_right)

    src = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [max_width-1, 0],
        [max_width-1, max_height-1],
        [0, max_height-1],
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(src, dst)
    # transformed = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return transform_matrix
