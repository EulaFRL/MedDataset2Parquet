from utils import *
from config import IMAGE_DIR, REPORT_DIR
import os
import pandas as pd

# @todo: Please revise this file and config.py according to how your data are stored.

data_X = {'images': [], 'report_text': [], 'metadata': [{}]}


# Function to handle multiple images in one patient_year folder
def process_image_folder(data_X, image_folder_path, report_text):
    image_files = sorted(os.listdir(image_folder_path))
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)

        # Check if the image is RGB and update metadata
        image_is_rgb(data_X, image_path)

        # @todo: 224x224, please check the function signature in utils.py for more information
        jpeg_to_np(data_X, 224, image_path)

        # report is repeated if it corresponds to multiple images
        data_X['report_text'].append(report_text)


image_folders = sorted(os.listdir(IMAGE_DIR))
report_files = sorted(os.listdir(REPORT_DIR))

for image_folder, report_file in zip(image_folders, report_files):
    image_folder_path = os.path.join(IMAGE_DIR, image_folder)
    report_path = os.path.join(REPORT_DIR, report_file)

    with open(report_path, 'r', encoding='utf-8') as file:
        report_text = file.read()

    process_image_folder(data_X, image_folder_path, report_text)


# @todo: Add some metadata
add_metadata(data_X, 'name', 'NLST')

# Convert to DataFrame and save as parquet
df = pd.DataFrame.from_dict(data_X)
df.to_parquet('data_X.parquet', index=False)