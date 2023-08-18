#!/usr/bin/env python3
"""Task 6"""

import cv2
import os
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
        """Non Max PLACEHOLDER"""
        selected_indices = []

        for i in np.unique(box_classes):
            idx = np.where(box_classes == i)
            class_filtered_boxes = filtered_boxes[idx]
            class_box_scores = box_scores[idx]
            keep = self._apply_nms(class_filtered_boxes, class_box_scores)

            selected_indices.extend(idx[0][keep])

        selected_boxes = filtered_boxes[selected_indices]
        selected_classes = box_classes[selected_indices]
        selected_scores = box_scores[selected_indices]

        return selected_boxes, selected_classes, selected_scores

    def _apply_nms(self, filtered_boxes, box_scores):
        """Applies Non Max Suppression"""
        sorted_indices = np.argsort(box_scores)[::-1]
        keep = []

        while sorted_indices.size > 0:
            best_box_idx = sorted_indices[0]
            keep.append(best_box_idx)
            remaining_indices = sorted_indices[1:]
            best_box = filtered_boxes[best_box_idx]
            remaining_boxes = filtered_boxes[remaining_indices]
            overlaps = self._calculate_iou(best_box, remaining_boxes)

            # Discard boxes with high IOU
            non_overlapping_indices = np.where(overlaps < self.nms_t)[0]
            sorted_indices = remaining_indices[non_overlapping_indices]

        return keep

    def _calculate_iou(self, box, boxes):
        """
        Calculate Intersection over Union between a box & a list of boxes.

        :param box: numpy.ndarray of shape (4,) containing the coordinates of
        the first box (x1, y1, x2, y2)
        :param boxes: numpy.ndarray of shape (n, 4) containing the coordinates
        of the n boxes to calculate IoU with
        :return: numpy.ndarray of shape (n,) containing the calculated IoUs
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iou = intersection_area / (
            box_area + boxes_area - intersection_area + 1e-9)
        return iou

    @staticmethod
    def load_images(folder_path):
        """
        Load images from the specified folder path.

        :param folder_path: a string representing the path
        to the folder holding all the images to load
        :return: a tuple of (images, image_paths)
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images by resizing and rescaling.

        :param images: a list of images as numpy.ndarrays
        :return: a tuple of (pimages, image_shapes)
        pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
        containing all of the preprocessed images
        ni: the number of images that were preprocessed
        input_h: the input height for the Darknet model - can vary by model
        input_w: the input width for the Darknet model - can vary by model
        3: number of color channels
        image_shapes: a numpy.ndarray of shape (ni, 2) containing the
        original height and width of the images
        2 => (image_height, image_width)
        """
        # Init empty lists to store preprocessed images & original shapes
        pimages = []  # Stores preprocessed images
        shapes = []  # Stores original shapes

        # Get the input dimensions from the model's input shape
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        # Loop through each image in the input list
        for i in images:
            # Get the original shape of the image
            img_shape = i.shape[0], i.shape[1]
            shapes.append(img_shape)  # Store the original shape
            # Resize the image using inter-cubic interpolation
            image = cv2.resize(i, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            # Normalize pixel values to the range [0, 1]
            image = image / 255.0  # Floating-point division
            # Append the preprocessed image to the pimages list
            pimages.append(image)

        # Convert the lists to numpy arrays
        pimages = np.array(pimages)  # Preprocessed images
        image_shapes = np.array(shapes)  # Original image shapes

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        # Draw bounding boxes and text on the image
        for i in range(len(boxes)):
            box = boxes[i]
            class_idx = box_classes[i]
            score = box_scores[i]

            # Draw bounding box
            color = (255, 0, 0)  # Blue color
            thickness = 2
            cv2.rectangle(image,
                          (int(box[0]),
                           int(box[1])),
                          (int(box[2]),
                           int(box[3])),
                          color,
                          thickness)

            # Draw class name and box score
            text = f"{self.class_names[class_idx]}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)  # Red color
            line_thickness = 1
            line_type = cv2.LINE_AA
            text_size = cv2.getTextSize(text,
                                        font,
                                        font_scale,
                                        line_thickness)[0]
            text_origin = (
                int(box[0]),
                int(box[1]) - 5)  # 5 pixels above the top left corner

            # Put text on the image
            cv2.putText(image,
                        text,
                        text_origin,
                        font,
                        font_scale,
                        text_color,
                        line_thickness,
                        line_type)

        # Display the image
        cv2.imshow(file_name, image)

        # Wait for a key press
        key = cv2.waitKey(0)

        # Check if 's' key is pressed to save the image
        if key == ord('s'):
            detections_folder = 'detections'
            if not os.path.exists(detections_folder):
                os.makedirs(detections_folder)
            output_path = os.path.join(detections_folder, file_name)
            cv2.imwrite(output_path, image)

        # Close the image window
        cv2.destroyAllWindows()
