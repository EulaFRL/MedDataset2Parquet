from utils import *
from config import POS_IMAGE_DIR, POS_REPORT_DIR, NEG_IMAGE_DIR, NEG_REPORT_DIR
import os
import pandas as pd

"""
main.py is designed for the following data storage form, PLEASE REVISE ACCORDING TO HOW YOUR DATASET IS STORED: 

In IMAGE_DIR, there are folders named by "pid_year", for example "100004_0". There could be one or multiple images in each folder. Reports are in REPORT_DIR, their names also starts with "pid_year", each report correspond to the image folder with the same pid_year, both image folders and reports are arranged in the right order, meaning that the images in the 1st image folder would correspond to the 1st report, and so on.

If there are multiple images in the same "pid_year" folder, the corresponding report is repeated to match the number of images. The metadata column contains an empty dictionary.

"""

data_X = {'images': [], 'report_text': [], 'label': [], 'metadata': [{}]}


# Function to handle multiple images in one patient_year folder
def process_image_folder(data_X, image_folder_path, report_text, label):
    image_files = sorted(os.listdir(image_folder_path))
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)

        # Check if the image is RGB and update metadata
        image_is_rgb(data_X, image_path)

        # @todo: originally stored in jpeg format
        # @todo: 224x224, please check the function signature in utils.py for more information
        jpeg_to_np(data_X, 224, image_path)

        # report is repeated if it corresponds to multiple images
        data_X['report_text'].append(report_text)

        data_X['label'].append(label)


# Process positive dataset
pos_image_folders = sorted(os.listdir(POS_IMAGE_DIR))
pos_report_files = sorted(os.listdir(POS_REPORT_DIR))

for image_folder, report_file in zip(pos_image_folders, pos_report_files):
    image_folder_path = os.path.join(POS_IMAGE_DIR, image_folder)
    report_path = os.path.join(POS_REPORT_DIR, report_file)

    with open(report_path, 'r', encoding='utf-8') as file:
        report_text = file.read()

    process_image_folder(data_X, image_folder_path, report_text, label=1)

# Process negative dataset
neg_image_folders = sorted(os.listdir(NEG_IMAGE_DIR))
neg_report_files = sorted(os.listdir(NEG_REPORT_DIR))

for image_folder, report_file in zip(neg_image_folders, neg_report_files):
    image_folder_path = os.path.join(NEG_IMAGE_DIR, image_folder)
    report_path = os.path.join(NEG_REPORT_DIR, report_file)

    with open(report_path, 'r', encoding='utf-8') as file:
        report_text = file.read()

    process_image_folder(data_X, image_folder_path, report_text, label=0)


# @todo: Add some metadata
add_metadata(data_X, 'name', 'NLST')

# Convert to DataFrame and save as parquet
df = pd.DataFrame.from_dict(data_X)
df.to_parquet('data_X.parquet', index=False)