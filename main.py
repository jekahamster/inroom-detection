import os
import pathlib
import math
import numpy as np
import cv2
from sklearn.compose import TransformedTargetRegressor
import utils
import cvzone
import cvyolo
import retina_detector

from defines import ROOT_DIR
# from yolo.yolo import Yolo


EPS = 1e-9
CIRCLE_RADIUS = 6
tracking_area_points = []


def _remove_point(point, radius=5):
    global tracking_area_points
    
    x, y = point
    for index, (_x, _y) in enumerate(tracking_area_points):
        if math.sqrt(math.pow(x - _x, 2) + math.pow(y - _y, 2)) < radius:
            tracking_area_points.pop(index)
            break


def _add_point(point, radius=5):
    global tracking_area_points

    print(point)
    x, y = point
    for index, (_x, _y) in enumerate(tracking_area_points):
        if math.sqrt(math.pow(x - _x, 2) + math.pow(y - _y, 2)) < radius:
            tracking_area_points.append((_x, _y))
            return
    tracking_area_points.append((x, y))


def mouse_callback(event, x, y, flags, param):
    global tracking_area_points
    
    point = (x, y)
    if event is cv2.EVENT_LBUTTONDOWN:
        _add_point(point, radius=CIRCLE_RADIUS)
        
    elif event is cv2.EVENT_RBUTTONDOWN:
        _remove_point(point, radius=CIRCLE_RADIUS)


def draw_tracking_area(image, radius=5):
    global tracking_area_points
    if not len(tracking_area_points):
        return

    for index, point in enumerate(tracking_area_points[:-1]):
        point2 = tracking_area_points[index+1]
        cv2.line(image, point, point2, (0, 255, 0), 2)

    for point in tracking_area_points:
        cv2.circle(image, center=point, radius=radius, color=(0, 255, 0), thickness=2)


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


def _rec_sizes(top_left, top_right, bottom_left, bottom_right):
    width_a = np.sqrt((top_left[0] - top_right[0])**2 + (top_left[1] - top_right[1])**2 )
    width_b = np.sqrt((bottom_left[0] - bottom_right[0])**2 + (bottom_left[1] - bottom_right[1])**2 )
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt((top_left[0] - bottom_left[0])**2 + (top_left[1] - bottom_left[1])**2 )
    height_b = np.sqrt((top_right[0] - bottom_right[0])**2 + (top_right[1] - bottom_right[1])**2 )
    max_height = max(int(height_a), int(height_b))
    return max_width, max_height, width_a, width_b, height_a, height_b


def get_perspective_matrix(image):
    global tracking_area_points

    top_left, top_right, bottom_left, bottom_right = sort_4_points(tracking_area_points)
    max_width, max_height, *_ = _rec_sizes(top_left, top_right, bottom_left, bottom_right)

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

def get_human_point(top_left, top_right, boom_right, bottom_left, transform_matrix):
    x1, y1 = top_left
    x2, y2 = top_right
    x3, y3 = boom_right
    x4, y4 = bottom_left
    # return int(x1 + x3) // 2, int(y1 + y3) // 2

    box_height = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    
    coords = np.array([[
        [(x4 + x3) // 2, y4 - box_height // 4]
    ]])

    transformed_coords = cv2.perspectiveTransform(coords, transform_matrix).squeeze().astype(np.int32)
    return transformed_coords


def show_perspective_view(image, transform_matrix, predictions):
    top_left, top_right, bottom_left, bottom_right = sort_4_points(tracking_area_points)
    
    width, height, *_ = _rec_sizes(top_left, top_right, bottom_left, bottom_right)
    transformered = cv2.warpPerspective(image, transform_matrix, (width, height))
    
    room_image = np.zeros((height, width, 1))
    for label, (x1, y1, x2, y2), score in predictions:
        w, h = x2 - x1, y2 - y1
        coords = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32)
        # hcoords = utils.to_homo_coords(coords)
        # transformered_coords = utils.from_homo_coords((transform_matrix @ hcoords.T).T)
        transformered_coords = cv2.perspectiveTransform(coords[np.newaxis, ...], transform_matrix)[0]
        (_x1, _y1), (_x2, _y2), (_x3, _y3), (_x4, _y4) = transformered_coords.astype("int32")
        # human_point = int(_x1 + _x3) // 2, int(_y1 + _y3) // 2
        # human_point = get_human_point((_x1, _y1), (_x2, _y2), (_x3, _y3), (_x4, _y4))

        human_point = get_human_point((x1, y1), (x2, y1), (x2, y2), (x1, y2), transform_matrix)

        cv2.line(transformered, (_x1, _y1), (_x2, _y2), (0, 255, 0), 1)
        cv2.line(transformered, (_x2, _y2), (_x3, _y3), (0, 255, 0), 1)
        cv2.line(transformered, (_x3, _y3), (_x4, _y4), (0, 255, 0), 1)
        cv2.line(transformered, (_x4, _y4), (_x1, _y1), (0, 255, 0), 1)

        cv2.circle(transformered, human_point, 5, (255, 255, 255), 1)
        cv2.circle(room_image, human_point, 5, 255, 1, cv2.LINE_AA)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    
    cv2.imshow("Transformed", transformered)
    cv2.imshow("Room", room_image)


def main():
    stop_frame = False
    show_perspective = False
    perspective_matrix = None
    fps_counter = cvzone.FPS()
    camera = cv2.VideoCapture(str(ROOT_DIR / "data" / "video1.mp4"))
    # model = Yolo(
    #     classes=str(ROOT_DIR / "yolo" / "model_data" / "coco" / "coco.names"),
    #     weights=str(ROOT_DIR / "yolo" / "model_data" / "yolov3.weights")
    # )

    model = cvyolo.CVYolo()
    # model = retina_detector.RetinaNetDetector()

    cv2.namedWindow('Window')
    cv2.setMouseCallback('Window', mouse_callback)

    frame = None
    while camera.isOpened():
        success, frame = camera.read() if not stop_frame else (True, frame)
        assert success, "Some problems with frame reading"
        
        frame = utils.proportional_resize(frame, 900) 
        out_frame = frame.copy()

        # predictions = [["person", (161, 83, 175, 94), 1], ["person", (448, 140, 472, 169), 1]]
        predictions = model.detect(frame)
        predictions = list(filter(lambda x: x[0] == "person", predictions))

        for label, (x1, y1, x2, y2), confidence in predictions:
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(out_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            print(label)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q"):
            break
        elif pressed_key == ord("f"):
            show_perspective = not show_perspective
            perspective_matrix = get_perspective_matrix(frame)
        elif pressed_key == ord("w"):
            stop_frame = not stop_frame
        
        if show_perspective:
            show_perspective_view(frame, perspective_matrix, predictions)

        fps, out_frame = fps_counter.update(out_frame, pos=(10, 20), color=(0, 255, 0), scale=1, thickness=1)
        draw_tracking_area(out_frame, radius=CIRCLE_RADIUS)
        cv2.imshow("Window", out_frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()