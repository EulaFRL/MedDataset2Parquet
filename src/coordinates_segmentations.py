import json
import os
import cv2
import numpy as np
import pandas as pd

"""
Originally for CDD-CESM; Could be used for segmentations of similar coordinate formats.
"""


def get_segmented_image(image_filenames, masks):

    img_masks = []
    for image_filename in image_filenames:
        image = cv2.imread(os.path.join(IMAGE_PATH,image_filename))
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


# @todo: change I/O paths
WRITE_PATH = ''
ANNOTATION_CSV_FILE = ''
IMAGE_PATH = ''

df = pd.read_csv(ANNOTATION_CSV_FILE)
try:
    os.makedirs(WRITE_PATH)
except:
    print("path already exists")



# generating binary masks&save
masks = df['region_shape_attributes'].tolist()
originals = df['#filename'].tolist()

for mask in get_segmented_image(originals, masks):
    GT_mask = np.array(mask) > 0
    if np.sum(GT_mask.astype(np.int)) == 0:
        continue

#@todo: directly write into parquet

# blend&save
# blended = alpha_blend(np.array(original), GT_mask.astype(np.int))
# cv2.imwrite(os.path.join(WRITE_PATH, images_names[batch_i]), blended)
