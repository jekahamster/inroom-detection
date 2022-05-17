import os
import pathlib
import math
import numpy as np
import cv2
import utils
import cvzone
import cvyolo
import retina_detector

from defines import ROOT_DIR
from main_window import MainWindow
from projective_window import ProjectiveWindow
from utils import sort_4_points
from utils import rect_sizes
from utils import get_perspective_matrix
# from yolo.yolo import Yolo


EPS = 1e-9
CIRCLE_RADIUS = 6


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


    frame = None
    frame_counter = 1
    main_window = MainWindow()
    perspective_window = None

    while camera.isOpened():
        success, frame = camera.read() if not stop_frame else (True, frame)
        assert success, "Some problems with frame reading"
        
        frame = utils.proportional_resize(frame, 900) 
        out_frame = frame.copy()

        if frame_counter % 30 != 0:
            frame_counter += 1
            continue
        else: 
            frame_counter = 1
            
        predictions = model.detect(frame)
        predictions = list(filter(lambda x: x[0] == "person", predictions))

        for label, (x1, y1, x2, y2), confidence in predictions:
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(out_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q"):
            break

        elif pressed_key == ord("f"):
            show_perspective = not show_perspective
            # perspective_matrix = get_perspective_matrix(main_window.tracking_area_points[:4])
            perspective_window = main_window.get_perspective_window()

        elif pressed_key == ord("w"):
            stop_frame = not stop_frame
        
        if show_perspective:
            # show_perspective_view(frame, perspective_matrix, predictions, main_window.tracking_area_points)
            perspective_window.show(predictions=predictions)

        fps, out_frame = fps_counter.update(out_frame, pos=(10, 20), color=(0, 255, 0), scale=1, thickness=1)
        main_window.show(out_frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()