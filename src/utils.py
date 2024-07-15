import os
import numpy as np
import pandas as pd
import cv2

"""
each parquet is a dataset
parquet format:
data_X = {'images':[image1, image2, ....],
'report_text':[report1, report2,...],
'label': [1, 1, ..., 0, 0, ...],
'metadata':[dict{}]}
"""


# Function to add key-value pairs to the metadata of parquet data_X
def add_metadata(data_X, key, value):
    data_X['metadata'][0][key] = value


# Function to transcribe jpeg image to 3D nested List(for compatibility with parquet) and add to data_X
def jpeg_to_nestedList(data_X, n, image_path, m=None):
    """
    :param data_X: the target parquet
    :param n: height, and width if m is not specified
    :param image_path: path/to/image
    :param m: width if specified, default to None
    Keep the original channels;
    (Optional) Raises an error if the original dimensions are smaller than the target dimensions;
    """

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unable to open: {image_path}")

        if m is None:
            m = n

        # Check image dimensions
        height, width = image.shape[:2]
        if width < n or height < m:
            raise ValueError("Image dimensions are smaller than the target dimensions")

        # Resize image
        image_np = cv2.resize(image, (n, m), interpolation=cv2.INTER_AREA)

        # Create the channel dimension if image is gray scale
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=0)
        else:
            image_np = image_np.transpose((2, 0, 1))

        # Convert the numpy array to a list of lists
        image_np_list = image_np.tolist()

        data_X['images'].append(image_np_list)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# Check if the images are RGB and update metadata
def image_is_rgb(data_X, path_to_image):
    try:
        image = cv2.imread(path_to_image)
        if image is None:
            raise ValueError(f"Image not found or unable to open: {path_to_image}")

        is_rgb = len(image.shape) == 3 and image.shape[2] == 3
        if 'is_rgb' not in data_X['metadata'][0]:
            data_X['metadata'][0]['is_rgb'] = is_rgb
    except Exception as e:
        print(f"Error checking if image is RGB {path_to_image}: {e}")


# Function to add report to data_X
def report_to_numpy(data_X, report_path):
    try:
        with open(report_path, 'r', encoding='utf-8') as file:
            report_text = file.read()

        data_X['report_text'].append(report_text)

    except Exception as e:
        print(f"Error processing report {report_path}: {e}")