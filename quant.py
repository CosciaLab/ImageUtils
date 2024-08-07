#!/usr/bin/env python
from loguru import logger
from skimage.measure import regionprops_table
import skimage.io as io
import numpy as np
import pandas as pd

import tifffile

import sys
import os
import argparse
from os.path import abspath
import time
import ast

try:
    from tqdm import tqdm   
except ImportError:
    logger.warning("tqdm not installed, install it for a progress bar")
    tqdm = lambda x: x

def get_args():

    description="""Easy-to-use, pixel quantification """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--image",    dest="image",       action="store", required=True, help="File path to input image or folder with images")
    inputs.add_argument("-l", "--label",    dest="label",       action="store", required=True, help="File path to input mask or folder with masks")
    inputs.add_argument("-m", "--markers",  dest="markers",     action="store", required=True, help="marker file path")
    inputs.add_argument("-o", "--output",   dest="output",      action="store", required=True, help="Path to output file or folder")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel", default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    inputs.add_argument("-w", "--math",     dest="math",        action="store", required=False,  type=str, nargs="*", default=['mean'],   help='Set the math operation (default: mean)')
    inputs.add_argument("-q", "--quantile", dest="quantile",    action="store", required=False,  type=str, nargs="*", default=None,     help='Set the quantile')
    
    arg = parser.parse_args()
    arg.image   = abspath(arg.image)
    arg.label   = abspath(arg.label)
    arg.markers = abspath(arg.markers)
    arg.output  = abspath(arg.output)
    return arg

def check_input_outputs(args):
    """ check if input is a file or a folder """
    # markers csv
    assert args.markers.endswith('.csv'), "Markers must be a csv file"
    assert os.path.isfile(args.markers), "Markers file must be a file"
    df = pd.read_csv(args.markers)
    expected_columns = {'channel_number', 'cycle_number', 'marker_name'}
    assert df.shape[1] >= 3, "Markers file must at least 3 columns"
    assert expected_columns.issubset(df.columns), f"DataFrame does not have the expected columns: {expected_columns}"
    assert df['channel_number'].nunique() == df.shape[0], "Channel numbers must be unique"
    assert df['marker_name'].nunique() == df.shape[0], "Cycle numbers must be unique"
    assert df['marker_name'].dtype == 'str' or "object", f"Marker name must be str, it is {df['marker_name'].dtype}"
    logger.info("Markers file checked")

    # check math
    logger.debug(f"Math provided {args.math}")
    assert isinstance(args.math, list), f"Math must be a list, you provided {args.math}"
    assert len(args.math) > 0, "Math list must have at least one element"
    expected_math = {'mean', 'max', 'min', 'std'} #set by match skimage.measure.regionprops_table, if median use quantile
    assert all(item in expected_math for item in args.math), f"Math operations must be in {expected_math}"
    logger.info(f"Math checked: {args.math}")

    # check quantile
    if args.quantile is not None:
        logger.debug(f"Quantile provided {args.quantile}")
        assert isinstance(args.quantile, list), f"Quantile must be a list, you provided {args.quantile}"
        assert len(args.quantile) > 0, "Quantile list must have at least one element"
        args.quantile = [float(string) for string in args.quantile]
        logger.info(f"Quantiles checked: {args.quantile}")

    # single files
    if os.path.isfile(args.image):
        assert os.path.isfile(args.label), "Both image and label must be files"
        assert args.image.endswith('.tif') and args.label.endswith('.tif'), "Both image and label must be tif files"
        assert args.markers.endswith('.csv'), "Markers must be a csv file"
        assert args.output.endswith('.csv'), "Output must be a csv file"
        logger.info("Inputs for single files checked")
        return "single_file"
    # directories
    elif os.path.isdir(args.image):
        assert os.path.isdir(args.label), "Both image and label must be folders"
        assert os.path.isfile(args.markers), "Markers must be a csv file"
        try:
            os.makedirs(args.output, exist_ok=True)
        except OSError:
            raise ValueError('Could not create output folder')
        logger.info("Inputs for folders checked")
        return "folder"
    else:
        raise ValueError('Input image and mask must match for being files or folders')

def check_image_and_labels(image_path:str, labels_path:str, markers_path:str):
    """ Check if image and labels are compatible for quantification """
    df = pd.read_csv(markers_path)
    with tifffile.TiffFile(image_path) as image, tifffile.TiffFile(labels_path) as labels:
            logger.info(f"Image shape is {image.series[0].shape}")
            logger.info(f"Image data type: {image.pages[0].dtype}")
            logger.info(f"Labels shape is {labels.series[0].shape}")
            logger.info(f"Labels data type: {labels.pages[0].dtype}")
            assert len(image.series[0].shape) == 3, "Image must have three dimensions"
            assert image.series[0].shape[0] == np.min(image.series[0].shape), "Image must be in c,y,x format"
            assert image.series[0].shape[0] == df.shape[0], "Number of markers and image channels do not match"
            assert image.series[0].shape[-2:] == labels.series[0].shape[-2:], "Image and labels must have the same shape"

def create_props(math:list):
    # calculate morphological properties
    props = ['centroid', 'area', 'axis_major_length', 'axis_minor_length','eccentricity','orientation', 'perimeter', 'solidity']
    for i in math:
        props.insert(0, f'intensity_{i}')
    logger.info(f"Calculating these {props} for each cell")
    return props

#median not available in skimage.measure, thus I create
def median(region_mask, intensity_image):
    values = intensity_image[region_mask]
    return np.median(values)

#to create a function per provided quantile
def create_quantile_function(quantile_value):
    def quantile_function(region_mask,intensity_image):
        values = intensity_image[region_mask]
        return np.quantile(a=values,q=quantile_value)
    setattr(quantile_function, "__name__", f"quantile{int(quantile_value * 100):02}")
    return quantile_function

def data_wrangling_and_cleanup(df, markers, quantiles):
    #create cell id column from index
    df.insert(0, "CellID", df.index)
    # defaults from skimage.measure.regionprops_table
    rename_map = {
        'centroid-1': 'X_centroid', 
        'centroid-0': 'Y_centroid', 
        'area': 'Area', 
        'axis_major_length': 'MajorAxisLength', 
        'axis_minor_length': 'MinorAxisLength', 
        'eccentricity': 'Eccentricity', 
        'solidity': 'Solidity', 
        'perimeter': 'Extent', 
        'orientation': 'Orientation'}
    #rename intensity columns
    list_of_intensity_columns = [col for col in df.columns if col.startswith("intensity")]
    for col in list_of_intensity_columns:
        parts = col.split('-')
        prefix = parts[0].split('intensity_')[1] 
        suffix = markers.at[int(parts[1]), "marker_name"] 
        rename_map[col] = prefix + "_" + suffix
    #rename quantiles columns if they exist
    if quantiles is not None:
        for col in [col for col in df.columns if col.startswith("quantile")]:
            parts = col.split('-')
            prefix = parts[0]
            suffix = markers.at[int(parts[1]), "marker_name"] 
            rename_map[col] = prefix + "_" + suffix
    #rename everything
    df.rename(columns=rename_map, inplace=True)
    # shift columns to the left of morphological columns
    index_Y_centroid    = df.columns.get_loc('Y_centroid')
    index_Solidity      = df.columns.get_loc('Solidity')
    columns_to_left     = list(df.columns[ : index_Y_centroid])
    columns_in_middle   = list(df.columns[index_Y_centroid : index_Solidity +1])
    columns_to_right    = list(df.columns[index_Solidity+1 : ])
    new_order = columns_to_left + columns_to_right + columns_in_middle
    df = df.reindex(columns=new_order)
    
    return df 

def quantify_single_file(image_path:str, labels_path:str, markers_path:str, output_path:str, props:list, quantiles:list):
    """ Quantify a single image and mask """
    check_image_and_labels(image_path, labels_path, markers_path)

    # load image, markers, and labels
    markers = pd.read_csv(markers_path)
    labeled_mask = io.imread(labels_path)
    multichannel_image = io.imread(image_path)

    # transpose image to x,y,c (expected from skimage)
    multichannel_image = np.transpose(multichannel_image, (1, 2, 0))
    logger.debug(f"Transposed multichannel shape: {multichannel_image.shape}; expect (x,y,c)")

    ## add quantiles to extra math operations as functions ##   
    extra_math = []
    if quantiles is not None:
        for q in quantiles:
            extra_math.append(create_quantile_function(q))

    ## Quantification ##
    properties = regionprops_table(
        label_image=labeled_mask, intensity_image=multichannel_image, 
        properties=props, extra_properties=extra_math)
    df = pd.DataFrame(properties)
    ## Data wrangling and cleanup ##
    df = data_wrangling_and_cleanup(df, markers, quantiles)
    #export
    df.to_csv(output_path, index=False)

def quantify_folder(image_path:str, labels_path:str, markers_path:str, output_path:str, props:list, quantiles:list):
    """ Quantify a folder of images and masks """
    list_of_files = [f for f in os.listdir(image_path) if f.endswith('.tif')]
    logger.info(f"Found {len(list_of_files)} files in the folder")

    for file in tqdm(list_of_files):
        logger.info(f"    Working on sample {file}")
        base_name = os.path.splitext(os.path.splitext(file)[0])[0]
        csv_file_path = os.path.join(output_path, base_name + '.csv')

        quantify_single_file(
            image_path=os.path.join(image_path, file), 
            labels_path=os.path.join(labels_path, file.split('.')[0] + '.tif'), 
            markers_path=markers_path, 
            output_path=csv_file_path,
            props=props,
            quantiles=quantiles
        )
        
def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    processing = check_input_outputs(args)
    props = create_props(args.math)

    if processing == "single_file":
        logger.info("Processing a single mask")
        quantify_single_file(args.image, args.label, args.markers, args.output, props, args.quantile)
    elif processing == "folder":
        logger.info("Processing a folder of masks")
        quantify_folder(args.image, args.label, args.markers, args.output, props, args.quantile)
    
    logger.success(f"Done, results saved in {args.output}")

if __name__ == "__main__":
    st = time.time()
    main()
    logger.success(f"Execution time: {time.time() - st:.1f} seconds ")

"""
Example usage:

python ./Jose_BI/ImageUtils/quant.py \
--image ./Jose_BI/data/mask_expansion_mesmer/dearray \
--label ./Jose_BI/data/mask_expansion_mesmer/segmentation_expansion \
--markers ./Jose_BI/data/mask_expansion_mesmer/markers.csv \
--output ./Jose_BI/data/mask_expansion_mesmer/quantification \
--log-level DEBUG --math 'mean' 'median' --quantile 0.75 0.85

"""