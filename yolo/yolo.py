import os
import pathlib
import numpy as np
import cv2
import time
import random
import tensorflow as tf
import colorsys

from yolo.yolov3.configs import YOLO_COCO_CLASSES, YOLO_V3_WEIGHTS
from yolo.yolov3.configs import YOLO_FRAMEWORK
from yolo.yolov3.configs import YOLO_INPUT_SIZE
from yolo.yolov3.utils import Load_Yolo_model, image_preprocess
from yolo.yolov3.utils import postprocess_boxes
from yolo.yolov3.utils import nms
from yolo.yolov3.utils import draw_bbox
from yolo.yolov3.utils import Load_Yolo_model
from yolo.yolov3.utils import read_class_names

def _get_bboxes(bboxes, CLASSES=YOLO_COCO_CLASSES):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    
    res = []
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        label = NUM_CLASS[class_ind]
        res.append( (label, (x1, y1, x2, y2), score) )
    
    return res


def _detect(Yolo, image, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45):
    times = []
    try:
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except:
        raise "Error" 

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    t1 = time.time()
    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
    
    t2 = time.time()
    
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    
    times.append(t2-t1)
    times = times[-20:]
    
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    
    # print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

    result = _get_bboxes(bboxes, CLASSES=CLASSES) 
    return result


class Yolo:
    def __init__(self, classes=YOLO_COCO_CLASSES, weights=YOLO_V3_WEIGHTS):
        self._model = Load_Yolo_model(classes=classes, weights=weights)
        self._classes = classes
        self._weights = weights

    def detect(self, image):
        return _detect(self._model, image, input_size=YOLO_INPUT_SIZE, CLASSES=self._classes)