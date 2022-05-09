import os
import pathlib
from unittest.mock import DEFAULT 
import numpy as np
import cv2

from defines import ROOT_DIR


DEFAULT_WEIGHTS_PATH = ROOT_DIR / "model_data" / "yolov3.weights"
DEFAULT_CONFIG_PATH = ROOT_DIR / "model_data" / "yolov3.cfg"
DEFAULT_CLASSES_PATH = ROOT_DIR / "model_data" / "coco.names"

class CVYolo:
    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, config_path=DEFAULT_CONFIG_PATH, classes_path=DEFAULT_CLASSES_PATH):
        self.classes = open(classes_path).read().strip().split('\n')

        self._net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image):
        layer_names = self._net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self._net.getUnconnectedOutLayers()]
        
        blob = cv2.dnn.blobFromImage(image, scalefactor=1./255., size=(416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outputs = self._net.forward(output_layers) # [channels, objects, coords+box_confidence+labels]

        boxes = []
        confidences = []
        classIDs = []
        h, w, *_ = image.shape

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        res = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                label = self.classes[classIDs[i]]
                res.append( (label, (x, y, x+w, y+h), confidence) )

        return res