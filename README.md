# MedDataset2Parquet

Parquet Format:
 data_X = {'images':[image1, image2, ...],
'segmentations': [seg4image1, seg4image2, ...],
'report_text':[report1, report2,...],
'label':[0, 1, ...],
'metadata':[dict{}]}

Segmentations are in the form of binary masks, optional.
label[i] = 0 means image i corresponds to a negative diagnosis, while 1 corresponds to a positive diagnosis.
Metadata are general information about the dataset.
Images are transcribed into serialized 3D numpy array.

Configs: POS_IMAGE_PATH, POS_REPORT_PATH, NEG_IMAGE_PATH, NEG_REPORT_PATH

main.py is designed for the following data storage form, PLEASE REVISE ACCORDING TO HOW YOUR DATASET IS STORED: 

Images are stored in JPEG format, transcribed into serialized 3D numpy array.

In IMAGE_DIR, there are folders named by "pid_year", for example "100004_0". There could be one or multiple images in each folder. Reports are in REPORT_DIR, their names also starts with "pid_year", each report correspond to the image folder with the same pid_year, both image folders and reports are arranged in the right order, meaning that the images in the 1st image folder would correspond to the 1st report, and so on.

If there are multiple images in the same "pid_year" folder, the corresponding report is repeated to match the number of images. The metadata column contains an empty dictionary.
