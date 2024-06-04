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

#TODO add meanmaxmin option in argparse
#TODO add for median
#TODO add for specific quantile
#TODO add for specific equation, expected 2D array of values


# redundancies
# TODO markers.csv columns order and names
# TODO images and masks should have to be exactly the same

# TODO either id system or samples sheet... 


# assumptions:
# images and masks should have the same name and shape
# not true for mesmer segmented images through MCMICRO

def get_args():

    description="""Easy-to-use, image quantification """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--image",    dest="image",       action="store", required=True, help="File path to input image or folder with images")
    inputs.add_argument("-l", "--label",    dest="label",       action="store", required=True, help="File path to input mask or folder with masks")
    inputs.add_argument("-m", "--markers",  dest="markers",     action="store", required=True, help="marker file path")
    inputs.add_argument("-o", "--output",   dest="output",      action="store", required=True, help="Path to output file or folder")
    inputs.add_argument("-ll", "--log-level", dest="loglevel", default='INFO', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help='Set the log level (default: INFO)')
    arg = parser.parse_args()

    arg.image   = abspath(arg.image)
    arg.label   = abspath(arg.label)
    arg.markers = abspath(arg.markers)
    arg.output  = abspath(arg.output)

    return arg


def check_input_outputs(args):
    """ check if input is a file or a folder """

    #check csv
    if args.markers.endswith('.csv'):
        df = pd.read_csv(args.markers, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#')
        assert df.shape[1] >= 3, "Markers file must have 3 columns"
        assert df.columns[0] == 'channel_number', "First column must be channel_number"
        assert df.columns[1] == 'cycle_number', "Second column must be cycle_number"
        assert df.columns[2] == 'marker_name', "Third column must be marker_name"
        assert df.dtypes[0] == 'int16', "First column must be int16"
        assert df.dtypes[1] == 'int16', "Second column must be int16"
        assert df.dtypes[2] == 'str' or "object", f"Third column must be str, it is {df.dtypes[2]}"
        assert df['marker_name'].nunique() == df.shape[0], "Marker names must be unique"
        logger.info(f"Markers file checked")

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
        #create output folder
        try:
            os.makedirs(args.output, exist_ok=True)
        except OSError:
            raise ValueError('Could not create output folder')
        logger.info(f"Inputs for folders checked")
        return "folder"
    
    else:
        raise ValueError('Input image and mask must match for being files or folders')
    
def quantify_single_file(image_path:str, labels_path:str, markers_path:str, output_path:str, meanmaxmin:bool = False):
    """ Quantify a single image and mask """
    # load markers
    markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#')
    logger.debug(f"Markers shape {markers.shape}")
    logger.debug(f"Markers columns {markers.columns}")
    logger.debug(f"Markers' markers {markers['marker_name'].tolist()}")
    # load images
    multichannel_image = io.imread(image_path)

    # cant assume (c,x,y), transpose if necessary
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
    #check max and min value of mask
    logger.debug(f"MASK: Max: {labeled_mask.max()} Min: {labeled_mask.min()}, 0_count: {np.count_nonzero(labeled_mask == 0)} 1_count: {np.count_nonzero(labeled_mask == 1)}")
    
    #check shapes of images and masks
    if multichannel_image.shape[2] != markers.shape[0]:
        raise ValueError("Number of Markers' markers and image channels do not match")
    if multichannel_image.shape[:-1] != labeled_mask.shape:
        raise ValueError("Image and labels must have the same shape")
    if np.unique(labeled_mask).shape[0] <= 2:
        raise ValueError("Labeled mask is binary, not labeled")
    
    # calculate morphological properties
    props = ['intensity_mean','area', 'axis_major_length', 'axis_minor_length','centroid','eccentricity','orientation', 'perimeter', 'solidity']
    properties = regionprops_table(label_image=labeled_mask, intensity_image=multichannel_image, properties=props)
    df = pd.DataFrame(properties)
    logger.info(f'Number of cells quantified : {df.shape[0]}')

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

    #list all columns that start as intensity_mean-*, the default from skimage.measure.regionprops_table
    list_of_channels = [col for col in df.columns if col.startswith('intensity_mean-')]
    #create map with new names
    channel_map = {col: col.split('intensity_mean-')[1] for col in list_of_channels}
    #edit the values of channel_map to match the names of the channels in the original image
    for key,value in channel_map.items():
        channel_map[key] = markers.at[int(value),"marker_name"]
    #rename columns
    df.rename(columns=channel_map, inplace=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Quantification results saved to {output_path}")

def quantify_folder(image_path:str, labels_path:str, markers_path:str, output_path:str, meanmaxmin:bool = False):
    """ Quantify a folder of images and masks """

    list_of_files = [f for f in os.listdir(image_path) if f.endswith('.tif')]
    logger.info(f"Found {len(list_of_files)} files in the folder")
    
    #assumes that the labels and images have the same name, most likely issues
    # how to handle if they dont have the same name, should have same beginning before dot

    for file in list_of_files:
        logger.info(f"    Working on sample {file}")
        # Remove the extension from the file
        output_without_extension = os.path.splitext(file)[0]
        csv_file_path = os.path.join(output_path, output_without_extension + '.csv')

        quantify_single_file(
            image_path=os.path.join(image_path, file), 
            labels_path=os.path.join(labels_path, file.split('.')[0] + '.tif'), 
            markers_path=markers_path, 
            output_path=csv_file_path,
            meanmaxmin = meanmaxmin)
        
def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    processing = check_input_outputs(args)
    if processing == "single_file":
        logger.info("Processing a single mask")
        quantify_single_file(args.image, args.label, args.markers, args.output, meanmaxmin=False)
    elif processing == "folder":
        logger.info("Processing a folder of masks")
        quantify_folder(args.image, args.label, args.markers, args.output, meanmaxmin=False)

if __name__ == "__main__":
    st = time.time()
    main()
    logger.info(f"Execution time: {time.time() - st:.1f} seconds ")


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