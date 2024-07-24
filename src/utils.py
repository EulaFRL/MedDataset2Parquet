import os
import numpy as np
import pandas as pd
import cv2
import json
import io

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


# jpeg images to seriealized 3d numpy array, for compatibility with parquet and with DataLoader
def jpeg_to_serialized_numpy(data_X, n, image_path, m=None):
    """
    :param data_X: the target data dictionary
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
        image_resized = cv2.resize(image, (m, n), interpolation=cv2.INTER_AREA)

        # Check if the image is grayscale or RGB
        if len(image_resized.shape) == 2:  # Grayscale image
            image_resized = np.expand_dims(image_resized, axis=0)  # Shape becomes (1, n, m)
        else:  # RGB image
            image_resized = image_resized.transpose((2, 0, 1))  # Shape becomes (3, n, m)

        # Serialize the numpy array
        buffer = io.BytesIO()
        np.save(buffer, image_resized)
        data_X['images'].append(buffer.getvalue())

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Check if the images are RGB and update metadata
def image_is_rgb(data_X, path_to_image):
    try:
        image = cv2.imread(path_to_image)
        if image is None:
            raise ValueError(f"Image not found or unable to open: {path_to_image}")

        is_rgb = len(image.shape) == 3 and image.shape[2] == 3
        # print(is_rgb)
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


def get_segmented_image(image_path, image_filenames, masks):

    img_masks = []
    for image_filename in image_filenames:
        image = cv2.imread(os.path.join(image_path, image_filename))
        img_masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))

    combined_masks = []
    for img_mask, mask in zip(img_masks, masks):
        if mask == '{}':
            continue
        mask = json.loads(mask)
        if mask['name'] == 'polygon':
            points = np.array(list(zip(mask['all_points_x'], mask['all_points_y'])), dtype=np.int32)
            cv2.fillPoly(img_mask, [points], 1)
        elif mask['name'] == 'ellipse' or mask['name'] == 'circle' or mask['name'] == 'point':
            if mask['name'] == 'circle':
                mask['rx'] = mask['ry'] = mask['r']
            elif mask['name'] == 'point':
                mask['rx'] = mask['ry'] = 25
            center = (int(mask['cx']), int(mask['cy']))
            axes = (int(mask['rx']), int(mask['ry']))
            cv2.ellipse(img_mask, center, axes, 0, 0, 360, 1, -1)

    return combined_masks
