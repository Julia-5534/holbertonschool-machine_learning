#!/usr/bin/env python3
"""Task 1"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """Yolo Class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for
        the Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the
        initial filtering step
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
        all of the anchor boxes:
            outputs: number of outputs (predictions) made by the Darknet model
            anchor_boxes: number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_height, img_width = image_size

        for output in outputs:
            grid_height, grid_width, num_anchors, _ = output.shape
            num_classes = output.shape[3] - 5

            # Initialize arrays for box calculations
            boxes_grid = np.zeros_like(
                output[..., :4])
            box_confidences_grid = np.zeros_like(
                output[..., 4:5])
            box_class_probs_grid = np.zeros_like(
                output[..., 5:])

            for i in range(grid_height):
                for j in range(grid_width):
                    for k in range(num_anchors):
                        t_x = output[i, j, k, 0]
                        t_y = output[i, j, k, 1]
                        t_w = output[i, j, k, 2]
                        t_h = output[i, j, k, 3]

                        # Apply sigmoid function
                        bx = 1 / (1 + np.exp(-t_x)) + j
                        by = 1 / (1 + np.exp(-t_y)) + i
                        bw = self.anchors[k, 0] * np.exp(t_w)
                        bh = self.anchors[k, 1] * np.exp(t_h)

                        # Normalize coordinates to image size
                        bx /= grid_width
                        by /= grid_height
                        bw /= img_width
                        bh /= img_height

                        x1 = (bx - bw / 2) * img_width
                        y1 = (by - bh / 2) * img_height
                        x2 = (bx + bw / 2) * img_width
                        y2 = (by + bh / 2) * img_height

                        boxes_grid[i, j, k] = [x1, y1, x2, y2]
                        box_confidences_grid[
                            i, j, k] = 1 / (1 + np.exp(
                                -output[i, j, k, 4]))
                        box_class_probs_grid[
                            i, j, k] = 1 / (1 + np.exp(
                                -output[i, j, k, 5:]))

            boxes.append(boxes_grid)
            box_confidences.append(box_confidences_grid)
            box_class_probs.append(box_class_probs_grid)

        return boxes, box_confidences, box_class_probs
