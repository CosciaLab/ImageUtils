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
try:
    from tqdm import tqdm   
except ImportError:
    logger.warning("tqdm not installed, install it for a progress bar")
    tqdm = lambda x: x

#TODO: Parallelize the processing of the masks

def get_args():
    """ Get arguments from command line """
    description = """Expand labeled masks by a certain number of pixels."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input", dest="input", action="store", required=True, help="File path to input mask or folders with many masks")
    inputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output mask, or folder where to save the output masks")
    inputs.add_argument("-p", "--pixels", dest="pixels", action="store", type=int, required=False, help="Image pixel size")
    inputs.add_argument("-l", "--log-level", dest="loglevel", default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
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
            if not args.input.endswith('.tif') or not args.output.endswith('.tif'):
                raise ValueError('Input and output files must be .tif files')
        return "single_file"
    
    elif os.path.isdir(args.input):
        logger.debug(f"Input is a folder")
        if os.path.isdir(args.output):
            logger.debug(f"Output is a folder")
            if not os.path.exists(args.input) or not os.path.exists(args.output):
                raise ValueError('Input and output folders must exist')
        
        #check if folders contain .tif files or folders
        list_of_files = [f for f in os.listdir(args.input) if f.endswith('.tif')]
        list_of_folders = [f for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, f))]
        logger.debug(f"Found {len(list_of_files)} files and {len(list_of_folders)} folders in the folder")

        assert len(list_of_files) > 0 or len(list_of_folders) > 0, "No .tif files or folders found in the input folder"

        if len(list_of_files) > 0 and len(list_of_folders) == 0:
            return "folder_of_tifs"
        elif len(list_of_files) == 0 and len(list_of_folders) > 0:
            return "folder_of_folders"
    
    else:
        raise ValueError('Input and output must match for being files or folders')
    
def expand_mask(input_path:str, output_path:str, how_many_pixels:int, type_of_input:str):

    """ Expand all masks in a folder by a certain number of pixels """
    logger.info(f"Processing folder {input_path} as {type_of_input}")

    if type_of_input == "single_file":
        label = io.imread(os.path.join(input_path))
        expanded_labels = segmentation.expand_labels(label, how_many_pixels)
        io.imsave(fname=os.path.join(output_path), arr=expanded_labels)

    elif type_of_input == "folder_of_tifs":
        list_of_files = [f for f in os.listdir(input_path) if f.endswith('.tif')]
        logger.debug(f"Found {len(list_of_files)} files in the folder")
        for file in tqdm(list_of_files):
            logger.info(f"    Working on sample {file}")
            label = io.imread(os.path.join(input_path, file))
            expanded_labels = segmentation.expand_labels(label, how_many_pixels)
            io.imsave(fname=os.path.join(output_path, file), arr=expanded_labels)

    elif type_of_input == "folder_of_folders":
        list_of_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
        logger.info(f"Found {len(list_of_folders)} subfolders in the folder")
        for folder in tqdm(list_of_folders):
            logger.info(f"    Working on subfolder {folder}")
            list_of_files = [f for f in os.listdir(os.path.join(input_path, folder)) if f.endswith('.tif')]
            assert len(list_of_files) > 0, f"No .tif files found in the subfolder {folder}"
            assert len(list_of_files) == 1, f"More than one .tif file found in the subfolder {folder}"
            label = io.imread(os.path.join(input_path, folder, list_of_files[0]))
            expanded_labels = segmentation.expand_labels(label, how_many_pixels)
            output_filename = folder.split('-')[1] + '.tif'
            io.imsave(fname=os.path.join(output_path, output_filename), arr=expanded_labels)

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    type_of_input = check_input_outputs(args)
    expand_mask(args.input, args.output, args.pixels, type_of_input=type_of_input)

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