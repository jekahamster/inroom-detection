import os 
import pathlib 
import numpy as np
import cv2 

from defines import ROOT_DIR
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption, label_color

DEFAULT_CLASSES_PATH = ROOT_DIR / "model_data" / "coco.names"
DEFAULT_MODEL_PATH = ROOT_DIR / "model_data" / "resnet50_coco_best_v2.1.0.h5"

class RetinaNetDetector:
    THRESHOLD = 0.5

    def __init__(self, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH):
        self.classes = open(classes_path).read().strip().split('\n')
        self._model = models.load_model(model_path, backbone_name="resnet50")
    
    def detect(self, image, threshold=THRESHOLD):
        image = preprocess_image(image)
        image, scale = resize_image(image)

        boxes, scores, labels = self._model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        res = []
        for box, score, label_index in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < threshold:
                break
            
            label = self.classes[label_index]
            x1, y1, x2, y2 = box.astype(np.int32)
            
            res.append( (label, (x1, y1, x2, y2), score) )
            
        return res
