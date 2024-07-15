import os
import numpy as np
import pandas as pd
from PIL import Image

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


# Function to transcribe jpeg image to 3D numpy array and add to data_X
def jpeg_to_np(data_X, n, image_path, m=None):
    """
    :param data_X: the target parquet
    :param n: height, and width if m is not specified
    :param image_path: path/to/image
    :param m: width if specified, default to None
    Keep the original channels;
    (Optional) Raises an error if the original dimensions are smaller than the target dimensions;
    """

    try:

        image = Image.open(image_path)
        if image.mode not in ("RGB", "L"):
            raise ValueError("Unsupported image mode: {}".format(image.mode))
        if m is None:
            m = n

        # Raise an error if the original dimensions are smaller than the target dimensions
        width, height = image.size
        if width < n or height < m:
            raise ValueError("Image dimensions are smaller than the target dimensions")

        image_np = np.array(image.resize((n, m), Image.ANTIALIAS))

        # Create the channel dimension if image is gray scale
        if image.mode == "L":
            image_np = np.expand_dims(image_np, axis=0)

        data_X['images'].append(image_np)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# Check if the images are RGB and update metadata
def image_is_rgb(data_X, path_to_image):
    try:
        image = Image.open(path_to_image)
        is_rgb = image.mode == "RGB"
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
