import os
import pathlib
import math
import numpy as np
import cv2
import utils


def get_human_point(top_left, top_right, boom_right, bottom_left, transform_matrix):
    x1, y1 = top_left
    x2, y2 = top_right
    x3, y3 = boom_right
    x4, y4 = bottom_left
    # return int(x1 + x3) // 2, int(y1 + y3) // 2

    box_height = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    
    coords = np.array([[
        # [(x4 + x3) // 2, y4 - box_height // 4]
        [(x4 + x3) // 2, y4]
    ]], dtype=np.float32)

    transformed_coords = cv2.perspectiveTransform(coords, transform_matrix).squeeze().astype(np.int32)
    return transformed_coords


class ProjectiveWindow:
    def __init__(self, transform_matrix, width, height, name="Projective Window"):
        self.name = name
        self.width = width
        self.height = height
        self.transform_matrix = transform_matrix

    def show(self, predictions=[]):
        room_image = np.zeros((self.height, self.width, 1))
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
            # transformered_coords = cv2.perspectiveTransform(coords[np.newaxis, ...], sel.ftransform_matrix)[0]
            # (_x1, _y1), (_x2, _y2), (_x3, _y3), (_x4, _y4) = transformered_coords.astype("int32")
            
            human_point = get_human_point((x1, y1), (x2, y1), (x2, y2), (x1, y2), self.transform_matrix)
            cv2.circle(room_image, human_point, 5, 255, 1, cv2.LINE_AA)
        
        cv2.imshow("Room", room_image)
