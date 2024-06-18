from __future__ import print_function, division
import time
import argparse
import os
import sys
import skimage
import tifffile

from loguru import logger
import pandas as pd
import numpy as np
import ome_types

try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert

# TODO scale to a folder of images

def get_args():
    description="""Projection of nuclear and membrane markers for segmentation"""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-m", "--markers",      dest="markers",     action="store", required=True, help="Channel metadata to be used for projection for nucleus")
    inputs.add_argument("-q", "--quantile",     dest="quantile",    action="store", required=True, type=float, help="Quantile value for thresholding")
    inputs.add_argument("-p", "--pixel-size",   dest="pixel_size",  action="store", required=False, type=float, default = None, help="pixel size in microns; default is 1.0")
    inputs.add_argument("-l", "--log-level",    dest="loglevel",    default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    
    arg = parser.parse_args()
    arg.input       = os.path.abspath(arg.input)
    arg.output      = os.path.abspath(arg.output)
    arg.markers     = os.path.abspath(arg.markers)
    arg.quantile    = float(arg.quantile)
    return arg

def check_inputs_and_outputs(args):

    #check paths
    assert os.path.isfile(args.input), "Input file does not exist"
    assert os.path.isfile(args.markers), "Marker file does not exist"

    #check markers
    assert args.markers.endswith('.csv'), "Markers file must be a .csv file"
    df = pd.read_csv(args.markers)
    logger.debug(f"Markers shape {df.shape}")
    logger.debug(f"Markers columns {df.columns}")
    logger.debug(f"Markers' markers {df['marker_name'].tolist()}")
    expected_columns = {'channel_number', 'cycle_number', 'marker_name', 'nuclear_markers', 'membrane_markers'}
    assert expected_columns.issubset(df.columns), f"DataFrame does not have the expected columns: {expected_columns}"
    assert df['channel_number'].nunique() == df.shape[0], "Channel numbers must be unique"
    assert df['marker_name'].nunique() == df.shape[0], "Cycle numbers must be unique"
    assert df['marker_name'].dtype == 'str' or "object", f"Marker name must be str, it is {df['marker_name'].dtype}"
    logger.info(f"Markers file checked")

    #check image path
    assert args.input.endswith('.tif'), "Input file must be a .tif file"

def load_image(image_path:str):
    logger.info(f"Loading image {image_path}")
    time_start = time.time()
    image = skimage.io.imread(image_path)
    logger.info(f"Loaded image with shape: {image.shape}, in {time.time()-time_start:.2f} seconds")
    return image

def separate_channels(image, markers_path:str):
    logger.info(f"Separating channels")
    
    df = pd.read_csv(markers_path)

    df['nuclear_markers'] = df['nuclear_markers'].notnull()
    df['membrane_markers'] = df['membrane_markers'].notnull()
    
    if image.shape[1] == image.shape[2]:
        logger.debug(f"Image is CYX")
    elif image.shape[0] == image.shape[1]:
        logger.debug(f"Image is YXC")
        image = np.moveaxis(image, 0, -1)
        logger.debug(f"Image shape is now {image.shape}")
    
    img_nuclear = image[df['nuclear_markers'],:,:]
    img_membrane = image[df['membrane_markers'],:,:]
    return img_nuclear, img_membrane

def detect_pixel_size(img_path, pixel_size=None):
    """ Detect pixel size from metadata """
    if pixel_size is None:
        try:
            metadata = ome_types.from_tiff(img_path)
            pixel_size = metadata.images[0].pixels.physical_size_x
        except Exception as err:
            print(err)
            print("Pixel size detection using ome-types failed")
            pixel_size = None
    return pixel_size

def process_nuclear_signal(image):
    logger.info(f"Processing nuclear signal")
    image = np.quantile(image, 0.75, axis=0)
    image = median_subtraction(image)
    image = skimage.filters.median(image, selem=skimage.morphology.disk(1.5))
    # min = 75%, max = 99%
    image = image - np.quantile(image, 0.75)
    image[image < 0] = 0
    image = image / np.quantile(image, 0.99)
    image[image > 1] = 1
    # scale to 0-255
    image = image * 255
    image = image.astype(np.uint8)
    return image

def process_membrane_signal(image, nucleus_img):
    image = np.quantile(image, 0.75, axis=0)
    image = median_subtraction(image)
    image = skimage.filters.median(image, skimage.morphology.disk(1.5))
    image[image < 0] = 0
    image = skimage.filters.maximum_filter(image, size=(3, 3))
    image = skimage.filters.median(image, skimage.morphology.disk(3))
    # min, max scaling
    image = image - np.quantile(image, 0.75)
    image[image < 0] = 0
    image = image / np.quantile(image, 0.995)
    image[image > 1] = 1
    image = image * 255
    image = image.astype(np.uint8)
    image = np.maximum(image, nucleus_img)
    return image

def median_subtraction(image):
    logger.info(f"Subtracting median")
    image_shape = image.shape
    new_shape = (image_shape[1]//8, image_shape[2]//8)
    #blur it before down sampling
    blur_img = skimage.filters.median(image, selem=skimage.morphology.disk(3.5))
    #downscale the median filtered image to the new dimensions
    downscaled_blur_image = skimage.transform.resize(blur_img, new_shape, order=0)
    #blur the small image
    blurred_small_image = skimage.filters.median(downscaled_blur_image, selem=skimage.morphology.disk(50.5))
    #scale back up
    image_blurred = skimage.transform.resize(blurred_small_image, image_shape, order=0)
    #remove blurred image from original image
    image_median_removed = image - np.minimum(image_blurred, image)
    return image_median_removed

def project_image(image, quantile:float):
    logger.info(f"Projecting image")
    time_start = time.time()
    projected_image = np.quantile(image, q=quantile, axis=0)
    logger.info(f"Projected image with shape: {projected_image.shape}, in {time.time()-time_start:.2f} seconds")
    return projected_image

def save_image(image, pixel_size, output_path:str):
    logger.info(f"Saving image to {output_path}")
    with tifffile.TiffWriter(output_path, ome=True, bigtiff=True) as tiff:
        tiff.write(
            data = image,
            shape = image.shape,
            dtype=image.dtype,
            metadata={'axes': 'CYX'},
            resolution=(10000 / pixel_size, 10000 / pixel_size, "centimeter"),
        )

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel)
    check_inputs_and_outputs(args)
    args.pixel_size = detect_pixel_size(args.input, args.pixel_size)
    image = load_image(args.input)
    img_nuclear, img_membrane = separate_channels(image, args.markers)
    projected_nuclear = project_image(img_nuclear, args.quantile)
    projected_membrane = project_image(img_membrane, args.quantile)
    projections = np.stack([projected_nuclear, projected_membrane], axis=0)
    save_image(projections, args.pixel_size, args.output)

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.success(f"Execution time: {time.time() - start_time:.1f} seconds ")