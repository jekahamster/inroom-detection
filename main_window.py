import os
import pathlib
import math
import numpy as np
import cv2
import utils
import color_constants as colors

from shapely.geometry import Point, Polygon
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
    window = param["window"]
    area_index = window.current_area_index
    
    if area_index == len(window.areas_points):
        window.areas_points.append([])
    
    tracking_area_points = window.areas_points[area_index]

    point = (x, y)
    if event is cv2.EVENT_LBUTTONDOWN:
        _add_point(point, tracking_area_points, radius=CIRCLE_RADIUS)
        
    elif event is cv2.EVENT_RBUTTONDOWN:
        _remove_point(point, tracking_area_points, radius=CIRCLE_RADIUS)

    while len(window.areas_points) > 1 and len(window.areas_points[-1]) == 0:
        window.areas_points.pop()
        window.current_area_index = len(window.areas_points)


def draw_tracking_area(image, points, color=colors.GREEN, radius=5):
    if not len(points):
        return image

    for index, point1 in enumerate(points[:-1]):
        point2 = points[index+1]
        cv2.line(image, point1, point2, color, 2)

    for point in points:
        cv2.circle(image, center=point, radius=radius, color=color, thickness=2)
        
    return image


def count_in_area(points, area_points):
    """
    points1 - inner points
    poitns2 - poly
    """
    cnt = 0
    polygon = Polygon(area_points)

    for x, y in points:
        point = Point((x, y))
        if point.within(polygon):
            cnt += 1

    return cnt


def _get_center(points):
    xs, ys = list(zip(*points))
    x = sum(xs) // len(xs)
    y = sum(ys) // len(ys)
    return x, y


class MainWindow:
    def __init__(self, name="MainWindow"):
        self.tracking_area_points = []
        self.name = name
        
        self.current_area_index = 0
        self.areas_points = [self.tracking_area_points]

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, mw_mouse_callback, param={
            "window": self
        })


    def show(self, image, predictions=[]):
        human_points = []
        for label, (x1, y1, x2, y2), label in predictions:
            point = ((x2+x1) // 2, y2)
            human_points.append(point)
            cv2.circle(image, center=point, radius=5, color=colors.AQUA, thickness=2)

        for index, points in enumerate(self.areas_points):
            color = colors.GREEN4

            if index == self.current_area_index:
                color = colors.GREEN

            if len(points) > 2 and points[0] == points[-1]:
                cnt_peoples_in_area = count_in_area(human_points, points)
                center_poly_x, center_poly_y = _get_center(points)
                cv2.putText(image, str(cnt_peoples_in_area), (center_poly_x, center_poly_y), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            image = draw_tracking_area(image, points, color=color)

        cv2.putText(image, str(self.current_area_index), (image.shape[1] - 25, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow(self.name, image)


    def get_perspective_window(self):
        top_left, top_right, bottom_left, bottom_right = utils.sort_4_points(self.tracking_area_points)
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


    def prev_area(self):
        if self.current_area_index > 0:
            self.current_area_index -= 1


    def next_area(self):
        if self.current_area_index < len(self.areas_points):
            self.current_area_index += 1
