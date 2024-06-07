#!/usr/bin/env python
import numpy
from skimage.measure import regionprops_table
import skimage.io as io
import numpy as np
import pandas as pd

import sys
import os
import argparse
from argparse import ArgumentParser as AP
from os.path import abspath
import time


# TODO add meanmaxmin option in argparse
# TODO add for median
# TODO add for specific quantile
# TODO add for specific equation, expected 2D array of values

def get_args():
    description = """Easy-to-use, image quantification """

    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--image", dest="image", action="store", required=True, help="File path to input image.")
    inputs.add_argument("-l", "--labels", dest="labels", action="store", required=True, help="File path to input mask.")
    inputs.add_argument("-m", "--markers", dest="markers", action="store", required=True, help="marker file path.")
    inputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output mask.")

    arg = parser.parse_args()

    arg.image = abspath(arg.image)
    arg.labels = abspath(arg.labels)
    arg.markers = abspath(arg.markers)
    arg.output = abspath(arg.output)

    return arg


def arg_check(image_path: str, labels_path: str, markers_path: str):
    # check if input_image_path is a tif
    if not image_path.endswith('.tif'):
        raise ValueError("Input image must be a tif")
    # check if input_labels_path is a tif
    if not labels_path.endswith('.tif'):
        raise ValueError("Input labels must be a tif")
    # check if input_markers_path is a csv
    if not markers_path.endswith('.csv'):
        raise ValueError("Input markers must be a csv")


def main(img, labels_path: str, markers_path: str):
    """
    Created by Jose Nimo

    Function:
    Calculates morphological and intensity properties of cells in a labeled image.
    Properties are calculated using skimage.measure.regionprops_table

    Variables:
    image_path = path to the image, expect multichannel tif, shape: (x, y, c)
    labels_path = path to the labels, expect single channel tif, shape: (x, y), with labeled mask, not binary
    markers_path = path to the markers, expect csv, with columns: ['channel_number', 'cycle_number','marker_name']

    Returns:
    dataframe object with morphological and intensity properties

    """

    # part 0, load markers
    try:
        markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#', sep=';')
    except:
        print('Table not ; delimited')
        pass

    try:
        markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#', sep=',')
    except:
        print('Could not read ' + markers_path)
        pass

    # part 1, load images
    multichannel_image = np.asarray(img)
    # multichannel_image = io.imread('/Users/r/Desktop/test/dearray/1.ome.tif')


    # cant assume (c,x,y), transpose if necessary
    if len(multichannel_image.shape) > 3:
        raise ValueError(f"Multichannel image must have 3 dimensions, {mutilchannel_image.shape} dimensions found")

    # if shape is c,x,y, transpose to x,y,c
    if multichannel_image.shape[0] < 100:
        # transpose is necessary
        multichannel_image = np.transpose(multichannel_image, (1, 2, 0))  # assumes going from x,y,c to c,x,y
        print("Transposing multichannel image from c,x,y to x,y,c")

    # part 2, load labels
    labeled_mask = io.imread(labels_path)  # Shape: (x, y)

    # QC
    print(f'multichannel_image shape {multichannel_image.shape}')
    print(f'markers shape {markers.shape}')
    print(f'labeled_mask shape {labeled_mask.shape}')

    # check shapes of images and masks
    if multichannel_image.shape[2] != markers.shape[0]:
        raise ValueError("Number of markers and channels do not match")
    if multichannel_image.shape[:-1] != labeled_mask.shape:
        raise ValueError("Image and labels must have the same shape")
    if np.unique(labeled_mask).shape[0] <= 2:
        raise ValueError("Labeled mask is binary, not labeled")

    # part 3, calculate morphological properties

    props = ['intensity_mean', 'area', 'axis_major_length', 'axis_minor_length', 'centroid', 'eccentricity',
             'orientation', 'perimeter', 'solidity']

    properties = regionprops_table(label_image=labeled_mask, intensity_image=multichannel_image, properties=props)
    df = pd.DataFrame(properties)

    print("Morhological properties calculated")
    print(f'Number of cells: {df.shape[0]}')

    # rename columns
    # create cell id column from index
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
        'orientation': 'Orientation'}

    df.rename(columns=rename_map, inplace=True)

    # rename channels from markers file
    # list all columns that start as intensity_mean-*, the default from skimage.measure.regionprops_table
    list_of_channels = [col for col in df.columns if col.startswith('intensity_mean-')]
    # create map with new names
    channel_map = {col: col.split('intensity_mean-')[1] for col in list_of_channels}
    # edit the values of channel_map to match the names of the channels in the original image
    for key, value in channel_map.items():
        channel_map[key] = markers.at[int(value), "marker_name"]
    # rename columns
    df.rename(columns=channel_map, inplace=True)

    return df


if __name__ == "__main__":
    args = get_args()

    st = time.time()
    # run check
    arg_check(args.image, args.labels, args.markers)
    # run script
    df = main(args.image, args.labels, args.markers, meanmaxmin=False)
    df.to_csv(args.output, index=False)
    rt = time.time() - st
    print(f"Script finished in {rt // 60:.0f}m {rt % 60:.0f}s")
    print(f"Quantification results saved to {args.output} \n ")