#!/usr/bin/env python3
"""Task 0"""

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
        boxes = []
        box_confidences = []
        box_class_probs = []

        img_height, img_width = image_size

        for output in outputs:
            grid_height, grid_width, num_anchors, _ = output.shape
            box = np.zeros(output[:, :, :, :4].shape)
            confidences = output[:, :, :, 4:5]
            class_probs = output[:, :, :, 5:]

            # Compute box coordinates relative to the original image
            for anchor_idx in range(num_anchors):
                box[:, :, anchor_idx, 0] = (
                    output[:, :, anchor_idx, 0] + self.anchors[
                        0, anchor_idx, 0]) / grid_width * img_width
                box[:, :, anchor_idx, 1] = (
                    output[:, :, anchor_idx, 1] + self.anchors[
                        0, anchor_idx, 1]) / grid_height * img_height
                box[:, :, anchor_idx, 2] = (np.exp(
                    output[:, :, anchor_idx, 2]) * self.anchors[
                        0, anchor_idx, 0]) / grid_width * img_width
                box[:, :, anchor_idx, 3] = (np.exp(
                    output[:, :, anchor_idx, 3]) * self.anchors[
                        0, anchor_idx, 1]) / grid_height * img_height

            boxes.append(box)
            box_confidences.append(confidences)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
