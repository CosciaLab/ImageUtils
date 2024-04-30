import pandas as pd
from pathlib import Path
import os

# Folder where segmentation is located
segmentations_path = Path(path) / 'segmentation'

# Folder where image is located
images_path = Path(path) / 'dearray'

markers_path = Path(path) / 'markers.csv'

# Folder to save new tif
image_output = os.path.join(path, 'median_subtracted')
if not os.path.exists(image_output):
    os.makedirs(image_output)

# Get a list of all files in the segmentation folder
segmentation_list = [file for file in segmentations_path.iterdir() if "mesmer-" in file.name]

# Create a DataFrame with the file paths
df = pd.DataFrame({'img_number': segmentation_list})

# Add the file name 'cell.tif' to each path
df['segmentation'] = df['img_number'].apply(lambda x: x / 'cell.tif')

# Get the basename of each path
df['img_number'] = df['img_number'].apply(lambda x: os.path.basename(x))

# Find "mesmer-" in the basename and replace it with an empty string
df['img_number'] = df['img_number'].str.replace('mesmer-', '')

# Get the basename of each path and combine with the extension
df['basename'] = df['img_number'].apply(lambda x: os.path.basename(x).replace('mesmer-', '') + '.ome.tif')

# Construct the file path for each row
df['image_input'] = df['basename'].apply(lambda x: os.path.join(images_path, x))

df['image_output'] = df['img_number'].apply(lambda x: os.path.join(image_output, x + '.tif'))
