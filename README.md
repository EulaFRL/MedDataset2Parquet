# MedDataset2Parquet

Parquet Format:
 data_X = {'instanceID':[pid_(year_slice/abnormalityID), ...],
'images':[image1, image2, ...],
'segmentations': [seg4image1, seg4image2, ...],
'report_text':[report1, report2,...],
'caption':[caption1, caption2,...],
'label':[0, 1, ...],
'metadata':[dict{}, repeated to match the length of other columns]}

Required fields: instanceID, images, at least one of report_text or caption, metadata

Optional fields: segmentations, label

Segmentations, if exists for the dataset, are in the form of binary masks in numpy array, serialized. Images are also in the form of serialized numpy arrays.
For negative images in a dataset with segmentations, the corresponding row in 'segementations' is replaced by 'None'
label[i] = 0 means image i corresponds to a negative diagnosis, while 1 corresponds to a positive diagnosis.
Metadata are general information about the dataset: required - 'name' 
Images are transcribed into serialized 3D numpy array.

Configs: POS_IMAGE_PATH, POS_REPORT_PATH, NEG_IMAGE_PATH, NEG_REPORT_PATH

utils.py contains helper functions.

PLEASE REVISE AND CREATE A SEPARATE main-DATASETNAME.py DEPENDING ON HOW YOUR DATA IS STORED.
