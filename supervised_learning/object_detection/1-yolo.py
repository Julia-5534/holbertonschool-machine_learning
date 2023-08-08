#!/usr/bin/env python3
"""Task 1"""

import numpy as np


class Yolo:
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
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        outputs: list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            grid_height & grid_width => height & width of grid used for output
            anchor_boxes => the number of anchor boxes used
            4 => (t_x, t_y, t_w, t_h)
            1 => box_confidence
            classes => class probabilities for all classes
        image_size: numpy.ndarray containing the image’s
        original size [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for
            each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box
                relative to original image
            box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively
            box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities for
            each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Process each output in outputs
        for i in range(len(outputs)):
            output = outputs[i]

            # Get grid dimensions and number of classes from output shape
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            num_classes = output.shape[3] - 5

            # Get anchor boxes for this output
            anchors = self.anchors[i]

            # Initialize arrays to store processed data for this output
            boxes_i = np.zeros((
                grid_height * grid_width * len(anchors), 4))
            box_confidences_i = np.zeros((
                grid_height * grid_width * len(anchors), 1))
            box_class_probs_i = np.zeros((
                grid_height * grid_width * len(anchors), num_classes))

            # Process each cell in the grid
            for row in range(grid_height):
                for col in range(grid_width):
                    # Get raw data from cell
                    raw_box_data = output[row][col][:, :4]
                    raw_box_confidence = output[row][col][:, [4]]
                    raw_box_class_probs = output[row][col][:, -num_classes:]

                    # Calculate processed data from raw data & store in arrays
                    boxes_i[
                        row*grid_width*len(anchors)+col*len(anchors):(
                            row*grid_width*len(anchors)+col*len(anchors)+len(
                                anchors)), :] = self._process_box_data(
                                    raw_box_data)
                    box_confidences_i[
                        row*grid_width*len(anchors)+col*len(anchors):(
                            row*grid_width*len(anchors)+col*len(anchors)+len(
                                anchors)), :] = raw_box_confidence
                    box_class_probs_i[
                        row*grid_width*len(anchors)+col*len(anchors):(
                            row*grid_width*len(anchors)+col*len(anchors)+len(
                                anchors)), :] = raw_box_class_probs

            # Reshape arrays to desired shape and append to lists
            boxes.append(
                np.reshape(boxes_i, (
                    grid_height, grid_width, len(anchors), 4)))
            box_confidences.append(
                np.reshape(box_confidences_i, (
                    grid_height, grid_width, len(anchors), 1)))
            box_class_probs.append(
                np.reshape(box_class_probs_i, (
                    grid_height, grid_width, len(anchors), num_classes)))

        return (boxes, box_confidences, box_class_probs)

    def _process_box_data(self, raw_box_data):
        """
        Helper method to process raw box data from a single cell in the grid
        raw_box_data: numpy.ndarray of shape (anchor_boxes, 4) containing the
        raw box data for a single cell in the grid
            4 => (t_x, t_y, t_w, t_h)
        Returns a numpy.ndarray of shape (anchor_boxes, 4) containing the
        processed box data for the cell
            4 => (x1, y1, x2, y2)
            (x1, y1, x2, y2) should represent the boundary box relative
            to original image
        """
        # TODO: Implement this method
        pass
