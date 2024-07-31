# Description: Create a projection of the nucleus and membrane channels for cellpose segmentation

import time
import tifffile
import os
from loguru import logger
import sys
import argparse
from pathlib import Path
import numpy as np
import ome_types

def get_args():
    """Get arguments from command line"""
    description = """Create a projection of the nucleus and membrane channels for cellpose segmentation"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-i", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-cn", "--channels_nucleus",     dest="channels_nucleus",    action="store", required=True, type=int, nargs='+', help="Channels to project for nucleus(0-based)")
    inputs.add_argument("-cm", "--channels_membrane",     dest="channels_membrane",    action="store", required=True, type=int, nargs='+', help="Channels to project for membrane(0-based)")
    inputs.add_argument("-nq", "--nuclear_quantile",    dest="nuclear_quantile",    action="store", required=True, type=float, help="Quantile value for nuclear projection")
    inputs.add_argument("-mq", "--membrane_quantile",   dest="membrane_quantile",    action="store", required=True, type=float, help="Quantile value for membrane projection")
    inputs.add_argument("-mmq", "--membrane_min_max_scaling_quantiles", dest="membrane_min_max_scaling_quantiles", action="store", required=False, type=str, nargs='+', help="Min and max quantiles for membrane projection")
    inputs.add_argument("-nmq", "--nucleus_min_max_scaling_quantiles",     dest="nucleus_min_max_scaling_quantiles",    action="store", required=False, type=str, nargs='+', help="Min and max quantiles for membrane projection")    
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')

    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    if arg.membrane_min_max_scaling_quantiles:
        arg.membrane_min_max_scaling_quantiles = [float(x) for x in arg.membrane_min_max_scaling_quantiles]
    if arg.nucleus_min_max_scaling_quantiles:
        arg.nucleus_min_max_scaling_quantiles = [float(x) for x in arg.nucleus_min_max_scaling_quantiles]
    return arg

def check_inputs_paths(args):
    """ check inputs and outputs """
    #input
    assert os.path.isfile(args.input), "Input file does not exist"
    assert args.input.endswith((".tif",".tiff")), "Input file must be a .tif or .tiff file"
    #output
    assert args.output.endswith((".tif",".tiff")), "Output file must be a .tif file"
    #channels_nucleus
    assert len(args.channels_nucleus) > 0, "No channels provided for nucleus"
    assert all([x >= 0 for x in args.channels_nucleus]), "All channel numbers must be positive"
    #channels_membrane
    assert len(args.channels_membrane) > 0, "No channels provided for membrane"
    assert all([x >= 0 for x in args.channels_membrane]), "All channel numbers must be positive"
    #nuclear quantile
    assert 0 <= args.nuclear_quantile <= 1, "Nuclear quantile must be between 0 and 1"
    #membrane quantile
    assert 0 <= args.membrane_quantile <= 1, "Membrane quantile must be between 0 and 1"
    #membrane min max quantiles
    if args.membrane_min_max_scaling_quantiles:
        assert len(args.membrane_min_max_scaling_quantiles) == 2, "Membrane min max quantiles must be a list of 2 values"
        assert all([0 <= float(x) <= 1 for x in args.membrane_min_max_scaling_quantiles]), "Membrane min max quantiles must be between 0 and 1"
    #nuclear min max quantiles
    if args.nucleus_min_max_scaling_quantiles:
        assert len(args.nucleus_min_max_scaling_quantiles) == 2, "Nuclear min max quantiles must be a list of 2 values"
        assert all([0 <= float(x) <= 1 for x in args.nucleus_min_max_scaling_quantiles]), "Nuclear min max quantiles must be between 0 and 1"
    #log level
    assert args.loglevel in ["DEBUG", "INFO"], "Log level must be either DEBUG or INFO"

def load_image(file_path):
    """Check if the file is an OME-TIFF."""
    file_path = Path(file_path)
    with tifffile.TiffFile(file_path) as tif:
        if tif.is_ome:
            logger.info(f"Image loaded is an OME-TIFF with metadata")
            return tif.asarray(), ome_types.from_tiff(file_path)
        else:
            logger.info(f"Image loaded is a TIFF without metadata")
            return tif.asarray(), None

def scale_2D_numpy_array(array, min_max_quantiles=[0.0, 1.0]):
    """Scale 2d numpy array to 8-bit with user chosen quantiles"""
    q_min = np.quantile(array, min_max_quantiles[0])
    q_max = np.quantile(array, min_max_quantiles[1])
    clipped_array = np.clip(array, q_min, q_max)
    rescaled_array = (clipped_array - q_min) / (q_max - q_min) * 255.0
    rescaled_array_8bit = rescaled_array.astype(np.uint8)
    return rescaled_array_8bit

def scale_image_channels(image, min_max_quantiles=[0.0, 1.0]):
    """Apply the scaling function to each channel of a (C, Y, X) image"""
    assert image.shape[0] == np.min(image.shape), "Image must with shape (C, Y, X)"
    scaled_channels = []
    for c in range(image.shape[0]):
        scaled_channel = scale_2D_numpy_array(image[c], min_max_quantiles)
        scaled_channels.append(scaled_channel)
    return np.stack(scaled_channels, axis=0)

def projection(image, channels, quantile=0.9, min_max_quantiles=[0.0, 1.0]):
    """Project the channels from a C,Y,X image"""
    logger.info(f"Creating projection for channels: {channels}")
    selected_channels_image = image[channels]
    scaled_image = scale_image_channels(selected_channels_image, min_max_quantiles)
    quantile_projection = np.quantile(scaled_image, quantile, axis=0).astype(np.uint8)
    return quantile_projection

def metadata_parse(ome_types_metadata=None):
    """Parse metadata from OME-TIFF"""
    pixel_size = ome_types_metadata.images[0].pixels.physical_size_x 
    metadata = {'axes': 'CYX','SignificantBits': 8,
        'PhysicalSizeX': pixel_size,'PhysicalSizeXUnit': "\u00b5m",
        'PhysicalSizeY': pixel_size,'PhysicalSizeYUnit': "\u00b5m",
        'Channel': {'Name': ['Nuclei_projection', 'Membrane_projection']}}
    return metadata

def main():
    args = get_args()
    check_inputs_paths(args)
    #logging setup
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    tif, metadata = load_image(args.input)
    nucleus_projection = projection(tif, args.channels_nucleus, args.nuclear_quantile, args.nucleus_min_max_scaling_quantiles)
    membrane_projection = projection(tif, args.channels_membrane, args.membrane_quantile, args.membrane_min_max_scaling_quantiles)
    concatenated_projection = np.stack([nucleus_projection, membrane_projection], axis=0)
    metadata = metadata_parse(metadata)
    logger.info(f"Saving projection")
    tifffile.imwrite(args.output, concatenated_projection, metadata=metadata)

if __name__ == "__main__":
    time_start = time.time()
    main()
    logger.info(f"Elapsed time: {int(time.time() - time_start)} seconds")


#example command
"""
python project.py \
--input /Users/jnimoca/Jose_BI/P26_SOPA_seg/data/image_data/991_subset.ome.tif \
--output /Users/jnimoca/Jose_BI/P26_SOPA_seg/data/projection_2.ome.tif \
--channels_nucleus 4 9 \
--channels_membrane 5 6 7 8 11 12 \
--nuclear_quantile 0.9 \
--membrane_quantile 0.9 \
-mmq 0.5 0.995 \
-nmq 0.4 0.990 
"""