#!/usr/bin/env python3
"""Task 3"""

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
        """
        Filter boxes based on their objectness score and class probabilities.

        :param boxes: list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 4) containing processed boundary boxes for each output,
        respectively
        :param box_confidences: list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 1) containing processed box confidences for
        each output, respectively
        :param box_class_probs: list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, classes) containing the processed box class
        probabilities for each output, respectively
        :return: tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            # Calculate the objectness score for each box
            obj_score = box_confidences[i] * box_class_probs[i]
            # Find the class with the highest score for each box
            max_scores = np.max(obj_score, axis=-1)
            max_classes = np.argmax(obj_score, axis=-1)
            # Filter out the boxes with a low objectness score
            mask = max_scores >= self.class_t
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        # Concatenate all the filtered boxes into a single array
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression on the boundary boxes.

        :param filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
        of the filtered bounding boxes
        :param box_classes: a numpy.ndarray of shape (?,) containing the class
        number for the class that filtered_boxes predicts, respectively
        :param box_scores: a numpy.ndarray of shape (?) containing the box
        scores for each box in filtered_boxes, respectively
        :return: tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        for c in set(box_classes):
            idxs = np.where(box_classes == c)
            filtered_boxes_c = filtered_boxes[idxs]
            box_scores_c = box_scores[idxs]
            box_classes_c = box_classes[idxs]
            idxs = K.backend.nms(filtered_boxes_c, box_scores_c,
                                 self.nms_t)
            box_predictions.append(filtered_boxes_c[idxs])
            predicted_box_classes.append(box_classes_c[idxs])
            predicted_box_scores.append(box_scores_c[idxs])
        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)
