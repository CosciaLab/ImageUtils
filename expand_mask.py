#system
from loguru import logger
import argparse
import sys
import os
from os.path import abspath
import time
#imports
import skimage.segmentation as segmentation
import skimage.io as io
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#TODO: Parallelize the processing of the masks
#TODO: Add a progress bar to show the progress of the processing
#TODO: some kind of renaming of the output files


def get_args():
    """ Get arguments from command line """
    description = """Expand labeled masks by a certain number of pixels."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input", dest="input", action="store", required=True, help="File path to input mask or folders with many masks")
    inputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output mask, or folder where to save the output masks")
    inputs.add_argument("-p", "--pixels", dest="pixels", action="store", type=int, required=False, help="Image pixel size")
    inputs.add_argument("-l", "--log-level", dest="loglevel", default='INFO', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help='Set the log level (default: INFO)')
    arg = parser.parse_args()
    # Standardize paths
    arg.input = abspath(arg.input)
    arg.output = abspath(arg.output)
    arg.pixels = int(arg.pixels)
    return arg


def check_input_outputs(args):
    """ check if input is a file or a folder """
    if os.path.isfile(args.input):
        if os.path.isfile(args.output):
            logger.info(f"Both input and output are files")
            #check if the input and output file is a .tif file
            if not args.input.endswith('.tif') or not args.output.endswith('.tif'):
                raise ValueError('Input and output files must be .tif files')
        return "single_file"
    
    elif os.path.isdir(args.input):
        if os.path.isdir(args.output):
            logger.info(f"Both input and output are folders")
            #check if the input and output folder exist
            if not os.path.exists(args.input) or not os.path.exists(args.output):
                raise ValueError('Input and output folders must exist')
        return "folder"
    
    else:
        raise ValueError('Input and output must match for being files or folders')
    
def batch_expand_mask(input_path:str, output_path:str, how_many_pixels:int):
    """ Expand all masks in a folder by a certain number of pixels """
    list_of_files = [f for f in os.listdir(input_path) if f.endswith('.tif')]
    logger.debug(f"Found {len(list_of_files)} files in the folder")
    for file in list_of_files:
        logger.info(f"    Working on sample {file}")
        label = io.imread(os.path.join(input_path, file))
        expanded_labels = segmentation.expand_labels(label, how_many_pixels)
        io.imsave(fname=os.path.join(output_path, file), arr=expanded_labels)

def expand_mask(input_path:str, output_path:str, how_many_pixels:int):
    """ Expand a single mask by a certain number of pixels """
    label = io.imread(input_path)
    expanded_labels = segmentation.expand_labels(label, how_many_pixels)
    io.imsave(fname=output_path, arr=expanded_labels)

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    processing = check_input_outputs(args)
    if processing == "single_file":
        logger.info("Processing a single mask")
        expand_mask(args.input, args.output, args.pixels)
    elif processing == "folder":
        logger.info("Processing a folder of masks")
        batch_expand_mask(args.input, args.output, args.pixels)

if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Execution time: {time.time() - start_time:.1f} seconds ")

"""
Example usages:

python expand_mask.py \
--log-level "DEBUG" \
--input "cylinter_demo/mask/" \
--output "cylinter_demo/output/" \
--pixels 5 

python expand_mask.py \
--log-level "DEBUG" \
--input "cylinter_demo/mask/15.tif" \
--output "cylinter_demo/output/15_expanded.tif" \
--pixels 5
"""