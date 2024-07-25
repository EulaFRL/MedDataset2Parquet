import pandas as pd
import os
import cv2
import numpy as np
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import io

"""
Test reading the first 10 rows of parquet files:
store segmentations and images separately into jpg, and the other rows into csv
"""

# Define paths
parquet_file = './NLST1.parquet'
output_image_dir = '../images&segmentations'
output_csv_file = 'remaining_rows.csv'

# Ensure the output image directory exists
os.makedirs(output_image_dir, exist_ok=True)

# Function to save numpy array as an image
def save_image_from_array(image_bytes, save_path):
    # Deserialize the numpy array from bytes
    image_np = np.load(io.BytesIO(image_bytes))
    # Check if the image is grayscale (shape should be (1, n, m)) or RGB (shape should be (3, n, m))
    if image_np.shape[0] == 1:
        # Grayscale image
        image_np = image_np.squeeze(axis=0)  # Shape becomes (n, m)
        image_np = (image_np * 255).astype(np.uint8)  # Ensure the type is uint8
    elif image_np.shape[0] == 3:
        # RGB image
        image_np = image_np.transpose((1, 2, 0))  # Shape becomes (n, m, 3)
        image_np = (image_np * 255).astype(np.uint8)  # Ensure the type is uint8
    else:
        raise ValueError(f"Invalid image array shape: {image_np.shape}")

    # Save the image using cv2
    cv2.imwrite(save_path, image_np)

# Read only the first 10 rows of the parquet file
pf = ParquetFile(parquet_file)
first_ten_rows = next(pf.iter_batches(batch_size=10))
first_10_rows = pa.Table.from_batches([first_ten_rows]).to_pandas()

# Process the first 10 rows and save the images
for index, row in first_10_rows.iterrows():
    image_path = os.path.join(output_image_dir, f'image_{index}.jpeg')
    save_image_from_array(row['images'], image_path)
    # @todo: with segmentations
    seg_path = os.path.join(output_image_dir, f'seg_{index}.jpeg')
    save_image_from_array(row['segmentations'], seg_path)

# Read the remaining columns of the parquet file
remaining_df = first_10_rows[['report_text', 'label', 'metadata']]

# Save the remaining columns to a CSV file
remaining_df.to_csv(output_csv_file, index=False)

print(f'First 10 rows images saved to {output_image_dir}')
print(f'Remaining rows saved to {output_csv_file}')