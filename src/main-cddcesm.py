import shutil

import numpy as np

from utils import *
from config import *
import os
import pandas as pd
import re

from spire.doc import *
from spire.doc.common import *


"""
modified main for CDD-CESM; contains coordinate segmentations in csv;
images were not originally separated into POS/NEG; separate by whether they have corresponding masks.
reports are for both breasts L/R, while there are different images for each;
reports are segmented into paragraphs for L/R, and stored accordingly.
"""

data_X = {'images': [], 'segmentations': [], 'report_text': [], 'label': [], 'metadata': [{}]}

print("Start parsing POS/NEG images")

image_files = sorted(filter_unwanted_files(os.listdir(ORIGINAL_IMAGE_DIR)))
report_files = sorted(filter_unwanted_files(os.listdir(ORIGINAL_REPORT_DIR)))


df = pd.read_csv(ANNOTATION_CSV_FILE)
# Create a mapping from filenames to masks(segmentation annotations) from the csv
filename_to_mask = dict(zip(df['#filename'], df['region_shape_attributes']))
# print(len(list(filename_to_mask.keys())))

# Separate negative images from positive images by whether they have corresponding masks
for file_name in image_files:
    if file_name not in filename_to_mask.keys():
        src_path = os.path.join(ORIGINAL_IMAGE_DIR, file_name)
        dest_path = os.path.join(NEG_IMAGE_DIR, file_name)

        if os.path.exists(src_path):
            # Move the file to the negative image directory
            shutil.move(src_path, dest_path)
            print(f"Moved to negative: {file_name}")
        else:
            print(f"Negative image not found: {file_name}")

print("Start parsing reports")

# @todo: parse reports into a dictionary with keys "pid_L/R

def extract_sections(text, type):

    match type:
        case 0: # First section
            pattern = re.compile(r'(?<=MAMMOGRAPHY REVEALED:)(.*?)(?=(Right Breast:|Left Breast:|OPINION:|$))', re.DOTALL)
        case 1: # Right breast
            pattern = re.compile(r'(?<=Right Breast:)(.*?)(?=(Left Breast:|OPINION'
                                 r'|CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:|$))', re.DOTALL)
        case 2: # Left breast
            pattern = re.compile(r'(?<=Left Breast:)(.*?)(?=(OPINION:'
                                 r'|CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:|$))', re.DOTALL)
        case _:
            raise(KeyError(f'No matching type: {type}'))

    matches = pattern.findall(text)

    # Filter matches to ensure no terminators or starters appear in the middle
    valid_matches = []
    for match in matches:
        if not re.search(r'(Right Breast:|Left Breast:|OPINION'
                         r'|CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:)', match[0]):
            valid_matches.append(match[0].strip())

    return "\n".join(valid_matches)

pidSide2report = {}
for report_file in report_files:

    pattern = re.compile(r'(\d*)(?=.docx)')
    pid = pattern.findall(report_file)[0]

    report_file = os.path.join(ORIGINAL_REPORT_DIR, report_file)
    document = Document()
    document.LoadFromFile(report_file)
    report_text = document.GetText()

    # print(report_text)

    start_text = extract_sections(report_text, 0)
    right_text = extract_sections(report_text, 1)
    left_text = extract_sections(report_text, 2)

    # print(f'____________________{pid}_______________________')
    # print(start_text)
    # print(right_text)
    # print(left_text)
    # print(f'____________________end______________________')

    if len(right_text) > 0:
        pidSide2report[pid+'R'] = (start_text, right_text)
    if len(left_text) > 0:
        pidSide2report[pid+'L'] = (start_text, left_text)

print("Start processing positive segmentations")

pos_image_files = sorted(filter_unwanted_files(os.listdir(ORIGINAL_IMAGE_DIR)))

masks = [filename_to_mask[filename] for filename in pos_image_files]   # masks written in coordinate form

for img_mask in get_segmented_image(ORIGINAL_IMAGE_DIR, pos_image_files, masks):
    # @todo: downsample to 224x224
    GT_mask = np.array(cv2.resize(img_mask, (224, 224), interpolation=cv2.INTER_AREA)) > 0    # turn into binary masks
    # Serialize the numpy array
    buffer = io.BytesIO()
    np.save(buffer, GT_mask)
    data_X['segmentations'].append(buffer.getvalue())

print("Start processing positive images and reports")

for image_file in pos_image_files:
    image_path = os.path.join(ORIGINAL_IMAGE_DIR, image_file)

    image_is_rgb(data_X, image_path)

    # @todo: originally stored in jpg format
    # @todo: 224x224, please check the function signature in utils.py for more information
    jpeg_to_serialized_numpy(data_X, 224, image_path)

    # search for the corresponding report
    # report is repeated if it corresponds to multiple images
    pattern = re.compile(r'(?<=P)(\d*)(?=_)')
    pid = pattern.findall(image_file)[0]
    pattern2 = re.compile(r'(?<=_)([L|R])')
    side = pattern2.findall(image_file)[0]

    report_text = pidSide2report[pid+side]
    if not report_text:
        raise FileNotFoundError(f"No corresponding report found for image file: {image_file}")
    report_text = "\n".join(report_text)  # @todo: for positive images

    data_X['report_text'].append(report_text)

    data_X['label'].append(1)

print("Start processing negative images and reports")

neg_image_files = sorted(filter_unwanted_files(os.listdir(NEG_IMAGE_DIR)))

for image_file in neg_image_files:
    image_path = os.path.join(NEG_IMAGE_DIR, image_file)

    image_is_rgb(data_X, image_path)

    # @todo: originally stored in jpg format
    # @todo: 224x224, please check the function signature in utils.py for more information
    jpeg_to_serialized_numpy(data_X, 224, image_path)

    # search for the corresponding report
    # report is repeated if it corresponds to multiple images
    pattern = re.compile(r'(?<=P)(\d*)(?=_)')
    pid = pattern.findall(image_file)[0]
    pattern2 = re.compile(r'(?<=_)([L|R])')
    side = pattern2.findall(image_file)[0]

    report_text = pidSide2report[pid + side]
    if not report_text:
        raise FileNotFoundError(f"No corresponding report found for image file: {image_file}")
    report_text = report_text[1]  # @todo: for negative images

    data_X['report_text'].append(report_text)
    data_X['segmentations'].append(None)
    data_X['label'].append(0)

print("Finished processing negative images and reports")

# @todo: Add some metadata
add_metadata(data_X, 'name', 'CDD-CESM')

# Convert to DataFrame and save as parquet
df = pd.DataFrame({
    'images': data_X['images'],
    'segmentations': data_X['segmentations'],
    'report_text': data_X['report_text'],
    'label': data_X['label'],
    'metadata': [data_X['metadata'][0]] * len(data_X['images'])  # Repeat metadata for each row
})

print("Converted to DataFrame, Start Saving as .parquet")

# @todo: change file name
df.to_parquet('CDD-CESM.parquet', index=False)