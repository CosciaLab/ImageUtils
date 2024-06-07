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

# Get a list of all files in the image folder
process_list = [file for file in images_path.iterdir() if ".ome.tif" in file.name]

# Create a DataFrame with the file paths
df = pd.DataFrame({'img_number': process_list})

# Get the basename of each path
df['img_number'] = df['img_number'].apply(lambda x: os.path.basename(x))

# Find "mesmer-" in the basename and replace it with an empty string
df['img_number'] = df['img_number'].str.replace('.ome.tif', '')

# Get the basename of each path and combine with the extension
df['basename'] = df['img_number'].apply(lambda x: os.path.basename(x) + '.ome.tif')

# Construct the file path for each row
df['image_input'] = df['basename'].apply(lambda x: os.path.join(images_path, x))
