import os
import pathlib
import math
import numpy as np
import cv2
import utils

from projective_window import ProjectiveWindow


CIRCLE_RADIUS = 5


def _remove_point(point, tracking_area_points, radius=5):
    x, y = point
    for index, (_x, _y) in enumerate(tracking_area_points):
        if math.sqrt(math.pow(x - _x, 2) + math.pow(y - _y, 2)) < radius:
            tracking_area_points.pop(index)
            break


def _add_point(point, tracking_area_points, radius=5):
    x, y = point
    for index, (_x, _y) in enumerate(tracking_area_points):
        if math.sqrt(math.pow(x - _x, 2) + math.pow(y - _y, 2)) < radius:
            tracking_area_points.append((_x, _y))
            return
    tracking_area_points.append((x, y))


def mw_mouse_callback(event, x, y, flags, param):
    tracking_area_points = param
    print("click")
    
    point = (x, y)
    if event is cv2.EVENT_LBUTTONDOWN:
        _add_point(point, tracking_area_points, radius=CIRCLE_RADIUS)
        
    elif event is cv2.EVENT_RBUTTONDOWN:
        _remove_point(point, tracking_area_points, radius=CIRCLE_RADIUS)


class MainWindow:
    def __init__(self, name="MainWindow"):
        self.tracking_area_points = []
        self.name = name
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, mw_mouse_callback, param=self.tracking_area_points)

    def show(self, image):
        self.draw_tracking_area(image, radius=CIRCLE_RADIUS)
        cv2.imshow(self.name, image)

    def draw_tracking_area(self, image, radius=5):
        if not len(self.tracking_area_points):
            return

        for index, point in enumerate(self.tracking_area_points[:-1]):
            point2 = self.tracking_area_points[index+1]
            cv2.line(image, point, point2, (0, 255, 0), 2)

        for point in self.tracking_area_points:
            cv2.circle(image, center=point, radius=radius, color=(0, 255, 0), thickness=2)

    def get_perspective_window(self):
        points = points[:4]
        top_left, top_right, bottom_left, bottom_right = utils.sort_4_points(points)
        max_width, max_height, *_ = utils.rect_sizes(top_left, top_right, bottom_left, bottom_right)

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
        return ProjectiveWindow(transform_matrix, max_width, max_height)