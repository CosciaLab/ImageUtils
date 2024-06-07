import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import binary_erosion
from scipy.ndimage import maximum_filter, minimum_filter

# Step 1: Read the TIFF images
masks_image_path = segmentation
nucleus_img_path = 'path/to/your/nucleus_image.tif'
cytosol_img_path = 'path/to/your/cytosol_image.tif'

masks = io.imread(masks_image_path)
nucleus_img = io.imread(nucleus_img_path)
cytosol_img = io.imread(cytosol_img_path)

new_cytosol_img = cytosol_img - nucleus_img
# Convert nucleus_img and cytosol_img to RGB and apply colors
nucleus_img_colored = np.stack([nucleus_img, np.zeros_like(nucleus_img), nucleus_img], axis=-1)  # Magenta
cytosol_img_colored = np.stack([np.zeros_like(new_cytosol_img), new_cytosol_img, np.zeros_like(new_cytosol_img)], axis=-1)  # Green

# Combine nucleus and cytosol images
combined_img = np.clip(nucleus_img_colored + cytosol_img_colored, 0, 255)

# Step 3: Create an outline mask by eroding and subtracting
# Step 3: Create an outline mask by eroding and subtracting
mask_extended = maximum_filter(masks, size=(3, 3))
mask_reduced = minimum_filter(masks, size=(10,10))
mask_difference = mask_reduced-mask_extended
mask_difference = mask_difference > 0
outline_masks = mask_difference.astype(np.uint8)

# Step 4: Create an overlay with yellow outlines
overlay_image = combined_img.copy()
overlay_image[outline_masks == 1] = [255, 255, 0]  # Yellow

# Display the result for verification
plt.figure(figsize=(10, 10))
plt.imshow(overlay_image)
plt.axis('off')
plt.show()

# Step 5: Save the resulting image as a PNG file
output_image_path = '/Users/rdeliza/Desktop/Multiplex/P21E14_HN46_restitch/segmentation/mesmer-15/overlay_image.png'
io.imsave(output_image_path, overlay_image.astype(np.uint8))
