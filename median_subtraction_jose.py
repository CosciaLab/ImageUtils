import os
import tifffile
import palom
import quant
import cv2 #pip install opencv-python

from skimage.filters import median, gaussian
from skimage.morphology import disk
import dask.array as da
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
# from cellpose import models, io
import matplotlib.pyplot as plt

# @ray.remote
def median_blur_remove(img):
    try:
        # Scale down 8x
        # Determine the new dimensions (1/8th of original size)
        old_width = img.shape[1]
        old_height = img.shape[0]
        new_width = old_width // 8
        new_height = old_height // 8

        # Blur it before down sampling
        med_blur_img = median(img, disk(3.5))
        # Resize the median filtered image to the new dimensions
        small_img = cv2.resize(med_blur_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # Blur small_image
        med_blur_small_img = median(small_img, disk(50.5))
        # Scale back up
        med_blur_small_img_fullres = cv2.resize(med_blur_small_img, (old_width, old_height))
        # Remove median
        med_rm = img - np.minimum(med_blur_small_img_fullres, img)
        return med_rm
    
    except:
        print('median_blur_remove error')
        pass


def process_nuclear_signal(image, markers_path):

    markers_table = pd.read_csv(markers_path)   
    markers_table = markers_table[~markers_table['marker_name'].str.contains('_bg')]

    nucleus_channels = markers_table[markers_table['marker_name'].str.contains('DAPI')]
    nucleus_channels = nucleus_channels.Channel_number - 1
    nucleus_img = original_image[nucleus_channels]
    print('Pulled from original_image')

    # Find 75th percentile for max
    nucleus_img = np.quantile(nucleus_img, 0.75, axis=0)
    print('Nuclei stains combined')

    nucleus_img = median_blur_remove(nucleus_img)
    nucleus_img = median(nucleus_img, disk(1.5))
    print('Median removed')

    nucleus_img = nucleus_img - np.quantile(nucleus_img, 0.75)
    nucleus_img[nucleus_img < 0] = 0
    print('Rescaled min')

    nucleus_img = nucleus_img / np.quantile(nucleus_img, 0.99)
    nucleus_img[nucleus_img > 1] = 1
    print('Rescaled max')

    nucleus_img = nucleus_img * 255
    nucleus_img = nucleus_img.astype(np.uint8)
    print('8-bit')

    tifffile.imwrite(os.path.join(segmentation_path, 'nucleus.tif'), nucleus_img, imagej=True, compression='lzw')
    except:
    print('Cannot create nucleus img')