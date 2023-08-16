#!/usr/bin/env python3
"""Task 2"""

import numpy as np
from tensorflow import keras as K


def sigmoid(x):
    """Sigmoid Function"""
    return 1 / (1 + np.exp(-x))


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
        """
        Process the outputs from the Darknet model for a single image.

        :param outputs: list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image
        :param image_size: numpy.ndarray containing the image’s original
        size [image_height, image_width]
        :return: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            # Get the coordinates
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            # Get the anchors
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            # Calculate the real coordinates
            bx = sigmoid(t_x) + np.arange(
                grid_width).reshape(1, grid_width, 1)
            by = sigmoid(t_y) + np.arange(
                grid_height).reshape(grid_height, 1, 1)
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)
            # Normalize the coordinates
            bx /= grid_width
            by /= grid_height
            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value
            # Calculate the coordinates relative to the image size
            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]
            # Update the box with the new coordinates
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            # Get the confidences and class probabilities
            box_confidence = sigmoid(output[..., 4])
            box_confidences.append(
                box_confidence.reshape(
                    grid_height, grid_width, anchor_boxes, 1))

            box_class_prob = sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter Boxes PLACEHOLDER"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            box = boxes[i]
            box_confidence = box_confidences[i]
            box_class_prob = box_class_probs[i]

            # box_score calculation
            box_score = box_confidence * box_class_prob
            box_classes_idx = np.argmax(box_score, axis=-1)
            box_class_scores = np.max(box_score, axis=-1)

            # Apply class score threshold
            mask = box_class_scores >= self.class_t
            filtered_box_scores = box_class_scores[mask]
            filtered_box_classes_idx = box_classes_idx[mask]
            filtered_box_coords = box[mask][..., :4][mask]

            filtered_boxes.extend(filtered_box_coords)
            box_classes.extend(filtered_box_classes_idx)
            box_scores.extend(filtered_box_scores)

        # Convert the lists to arrays after the loop
        filtered_boxes = np.array(filtered_boxes)
        box_classes = np.array(box_classes)
        box_scores = np.array(box_scores)

        return filtered_boxes, box_classes, box_scores
