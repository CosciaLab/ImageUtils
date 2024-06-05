#!/usr/bin/env python
from loguru import logger
from skimage.measure import regionprops_table
import skimage.io as io
import numpy as np
import pandas as pd

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

#TODO add for median
#TODO add for specific quantile
#TODO add for specific equation, expected 2D array of values


def get_args():

    description="""Easy-to-use, pixel quantification """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--image",    dest="image",       action="store", required=True, help="File path to input image or folder with images")
    inputs.add_argument("-l", "--label",    dest="label",       action="store", required=True, help="File path to input mask or folder with masks")
    inputs.add_argument("-m", "--markers",  dest="markers",     action="store", required=True, help="marker file path")
    inputs.add_argument("-o", "--output",   dest="output",      action="store", required=True, help="Path to output file or folder")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel", default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    inputs.add_argument("-w", "--math",     dest="math",        action="store", required=False,  type=str, nargs="*", default=['mean'], help='Set the math operation (default: mean)')
    # inputs.add_argument("-q", "--quantile", dest="quantile",    action="store", required=False,  type=str, nargs="*", default=None, help='Set the quantile (default: 0.5)')
    
    arg = parser.parse_args()
    arg.image   = abspath(arg.image)
    arg.label   = abspath(arg.label)
    arg.markers = abspath(arg.markers)
    arg.output  = abspath(arg.output)
    return arg


# expand skimage.measure to include median



def check_input_outputs(args):
    """ check if input is a file or a folder """

    #check csv
    if args.markers.endswith('.csv'):
        df = pd.read_csv(args.markers)
        logger.debug(f"Markers shape {df.shape}")
        logger.debug(f"Markers columns {df.columns}")
        logger.debug(f"Markers' markers {df['marker_name'].tolist()}")
        expected_columns = {'channel_number', 'cycle_number', 'marker_name'}
        assert df.shape[1] >= 3, "Markers file must at least 3 columns"
        assert expected_columns.issubset(df.columns), f"DataFrame does not have the expected columns: {expected_columns}"
        assert df['channel_number'].nunique() == df.shape[0], "Channel numbers must be unique"
        assert df['marker_name'].nunique() == df.shape[0], "Cycle numbers must be unique"
        assert df['marker_name'].dtype == 'str' or "object", f"Marker name must be str, it is {df['marker_name'].dtype}"
        logger.info(f"Markers file checked")

    #check math
    logger.debug(f"Math provided {args.math}")
    assert isinstance(args.math, list), f"Math must be a list, you provided {args.math}"
    assert len(args.math) > 0, "Math list must have at least one element"
    expected_math = {'mean', 'max', 'min', 'median', 'mode', 'std'}
    assert all(item in expected_math for item in args.math), f"Math operations must be in {expected_math}"

    #check quantile
    # assert isinstance(args.quantile, list), f"Quantile must be a list, you provided {args.quantile}"
    # if len(args.quantile) > 0:
    #     assert all(isinstance(item, float) for item in args.quantile), "All elements of quantile must be floats"

    #check image and mask
    if os.path.isfile(args.image):
        assert os.path.isfile(args.label), "Both image and label must be files"
        assert args.image.endswith('.tif') and args.label.endswith('.tif'), "Both image and label must be tif files"
        assert args.markers.endswith('.csv'), "Markers must be a csv file"
        assert args.output.endswith('.csv'), "Output must be a csv file"
        logger.info(f"Inputs for single files checked")
        return "single_file"

    elif os.path.isdir(args.image):
        assert os.path.isdir(args.label), "Both image and label must be folders"
        assert os.path.isfile(args.markers), "Markers must be a csv file"
        try:
            os.makedirs(args.output, exist_ok=True)
        except OSError:
            raise ValueError('Could not create output folder')
        logger.info(f"Inputs for folders checked")
        return "folder"
    
    else:
        raise ValueError('Input image and mask must match for being files or folders')

def create_props(math:list):
    # calculate morphological properties
    props = ['centroid', 'area', 'axis_major_length', 'axis_minor_length','eccentricity','orientation', 'perimeter', 'solidity']
    for i in math:
        props.insert(0, f'intensity_{i}')
    logger.info(f"Calculating these {props} for each cell")
    return props

def quantify_single_file(image_path:str, labels_path:str, markers_path:str, output_path:str, props:list):
    """ Quantify a single image and mask """
    markers = pd.read_csv(markers_path)
    multichannel_image = io.imread(image_path)

    if len(multichannel_image.shape) != 3:
        raise ValueError(f"Multichannel image must have 3 dimensions, this shape {multichannel_image.shape} found")
    
    # if shape is c,x,y, transpose to x,y,c, expected from skimage
    logger.debug(f"Multichannel image shape {multichannel_image.shape}")
    if multichannel_image.shape[0] < multichannel_image.shape[2]:
        multichannel_image = np.transpose(multichannel_image, (1, 2, 0))
        logger.debug(f"Transposed multichannel shape: {multichannel_image.shape}, expect (x,y,c)")
    
    # load labels
    labeled_mask = io.imread(labels_path)
    logger.debug(f"Labeled mask shape {labeled_mask.shape}")
    logger.debug(f"MASK: Max: {labeled_mask.max()}, Min: {labeled_mask.min()}, 0_count: {np.count_nonzero(labeled_mask == 0)}, 1_count: {np.count_nonzero(labeled_mask == 1)}")
    
    #check shapes of images and masks
    if multichannel_image.shape[2] != markers.shape[0]:
        raise ValueError("Number of Markers' markers and image channels do not match")
    if multichannel_image.shape[:-1] != labeled_mask.shape:
        raise ValueError("Image and labels must have the same shape")
    if np.unique(labeled_mask).shape[0] <= 2:
        raise ValueError("Labeled mask is binary, not labeled")

    # pass on median as extra function
    def median(region_mask, intensity_image):
        values = intensity_image[region_mask]
        return np.median(values)

    extra_math = []
    if "intensity_median" in props:
        extra_math.append(median)
        props = [s for s in props if s != 'intensity_median']
        rename_median = True

    properties = regionprops_table(label_image=labeled_mask, intensity_image=multichannel_image, properties=props, extra_properties=extra_math)
    df = pd.DataFrame(properties)

    #create cell id column from index
    df.insert(0, "CellID", df.index)

    rename_map = {
        'centroid-1': 'X_centroid', 
        'centroid-0': 'Y_centroid', 
        'area': 'Area', 
        'axis_major_length': 'MajorAxisLength', 
        'axis_minor_length': 'MinorAxisLength', 
        'eccentricity': 'Eccentricity', 
        'solidity': 'Solidity', 
        'perimeter': 'Extent', 
        'orientation': 'Orientation'
    }
    df.rename(columns=rename_map, inplace=True)

    list_of_intensity_columns = [col for col in df.columns if col.startswith("intensity")]
    # Simplify the original line of code
    column_rename_map = {}
    for col in list_of_intensity_columns:
        # I expect names as: intensity_mean-1, intensity_mean-2, etc.
        parts = col.split('-')
        #get math name
        prefix = parts[0].split('intensity_')[1] 
        #get marker name
        suffix = markers.at[int(parts[1]), "marker_name"] 
        new_col_name = prefix + "_" + suffix
        # Add to the rename map
        column_rename_map[col] = new_col_name

    if rename_median == True:
        #median columns called median-0, median-1, median-2, etc.
        column_median_rename_map = {}
        for col in [col for col in df.columns if col.startswith("median-")]:
            parts = col.split('-')
            prefix = parts[0]
            suffix = markers.at[int(parts[1]), "marker_name"] 
            new_col_name = prefix + "_" + suffix
            column_median_rename_map[col] = new_col_name
        
        df.rename(columns=column_median_rename_map, inplace=True)
        
        # shift columns to the left of morphological columns
        index_Y_centroid = df.columns.get_loc('Y_centroid')
        columns_to_left = df.columns[ : index_Y_centroid]
        index_Solidity = df.columns.get_loc('Solidity')
        columns_to_right = df.columns[index_Solidity+1 : ]
        columns_in_middle = df.columns[index_Y_centroid : index_Solidity +1]
        new_order = columns_to_left + columns_to_right + columns_in_middle
        df = df.reindex(columns=new_order)

    df.rename(columns=column_rename_map, inplace=True)
    df.to_csv(output_path, index=False)

def quantify_folder(image_path:str, labels_path:str, markers_path:str, output_path:str, props:list):
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
            props=props
        )
        
def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    processing = check_input_outputs(args)
    props = create_props(args.math)

    if processing == "single_file":
        logger.info("Processing a single mask")
        quantify_single_file(args.image, args.label, args.markers, args.output, props)
    elif processing == "folder":
        logger.info("Processing a folder of masks")
        quantify_folder(args.image, args.label, args.markers, args.output, props)
    
    logger.success(f"Done, results saved in {args.output}")

if __name__ == "__main__":
    st = time.time()
    main()
    logger.success(f"Execution time: {time.time() - st:.1f} seconds ")


"""
Example usage:

python quant.py \
--log-level "DEBUG" \
--image cylinter_demo/tif/1.ome.tif \
--label cylinter_demo/mask/1.tif \
--markers cylinter_demo/markers.csv \
--output cylinter_demo/quantified.csv

python quant.py \
--log-level "DEBUG" \
--image cylinter_demo/tif/ \
--label cylinter_demo/mask/ \
--markers cylinter_demo/markers.csv \
--output cylinter_demo/quantification/

"""