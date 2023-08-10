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
        """Processes outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_height, img_width = image_size

        for output in outputs:
            grid_height, grid_width, num_anchors, _ = output.shape
            num_classes = output.shape[3] - 5

            grid_y = np.arange(grid_height).reshape(1, grid_height, 1, 1)
            grid_x = np.arange(grid_width).reshape(1, 1, grid_width, 1)

            anchor_width = self.anchors[..., 0]
            anchor_height = self.anchors[..., 1]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            box_confidence = output[..., 4]
            box_class_probs = output[..., 5:]

            # Apply sigmoid function to bounding box parameters
            t_x = 1 / (1 + np.exp(-t_x))
            t_y = 1 / (1 + np.exp(-t_y))
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_probs = 1 / (1 + np.exp(-box_class_probs))

            pred_boxes_x = (t_x + grid_x) / grid_width
            pred_boxes_y = (t_y + grid_y) / grid_height
            pred_boxes_w = anchor_width * np.exp(t_w)
            pred_boxes_h = anchor_height * np.exp(t_h)

            # Normalize the box coordinates to the image size
            pred_boxes_x *= img_width
            pred_boxes_y *= img_height
            pred_boxes_w *= img_width
            pred_boxes_h *= img_height

            # Append results to respective lists
            boxes.append(np.stack([
                pred_boxes_x,
                pred_boxes_y,
                pred_boxes_w,
                pred_boxes_h], axis=-1))
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_probs)

        return boxes, box_confidences, box_class_probs
