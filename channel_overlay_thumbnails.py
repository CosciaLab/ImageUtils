# Description: Create PNG overlays of specific channels from tif
import numpy as np
import skimage
import matplotlib.pyplot as plt
import argparse
import os
import sys
from loguru import logger
import ast
import natsort
import tqdm
from skimage import exposure

#TODO naturalsort the files in the directory
#TODO scale image downsize according to image size

def get_args():
    """Get arguments from command line"""
    description = """Create PNG overlays of specific channels from tif"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-c", "--channels",     dest="channels",    action="store", required=True, type=str, help="Channels to overlay (0-based)")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    default='INFO', choices=["DEBUG", "INFO"], help='Set the log level (default: INFO)')
    
    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    arg.output = os.path.abspath(arg.output)
    return arg

#check if input and output are for single file or for directories
def check_inputs_paths(args):
    """
    Check if input and output are for single file or for directories
    """
    if os.path.isfile(args.input):
        logger.info("Input is a single file")
        assert args.output.endswith(".png"), "Output file must be a PNG file"
        return "single_file"
        
    elif os.path.isdir(args.input):
        logger.info("Input is a directory")
        if os.path.isdir(args.output):
            logger.info("Output is also a directory")
            return "directory"
        else:
            logger.error("Input is a directory, and output is not a directory")
            sys.exit(1)

    else:
        logger.error("Input does not pass the os.path.isfile or os.path.isdir test")
        sys.exit(1)

def process_numbers(numbers_str):
    # Convert the string representation of list to a Python list
    numbers = ast.literal_eval(numbers_str)
    return numbers

def colorize(im, color, clip_percentile=0.5):
    """Helper function to create an RGB image from a single-channel image using a specific color."""

    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError('This function expects a single-channel image!')
    
    # Contrast stretching
    p2, p98 = np.percentile(im, (2, 98))
    im_rescaled = exposure.rescale_intensity(im, in_range=(p2, p98))

    # Rescale the image according to how we want to display it
    im_scaled = im_rescaled.astype(np.float32) - np.percentile(im_rescaled, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    im_scaled = np.clip(im_scaled, 0, 1)
    
    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)
    
    # Reshape the color (here, we assume channels last)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color

def create_channel_labels(channels):
    return {channel: f'Channel {channel + 1}' for channel in channels}

def get_overlay(input, output, channels):
    """ Create overlay of specific channels from tif """
    image = skimage.io.imread(input)
    image_shape = image.shape
    sample_name = os.path.basename(input).split(".")[0]

    if len(channels) > 7:
        logger.error("Too many channels to overlay, max is 7")
        sys.exit(1)
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    for channel,color in zip(channels,colors):
        ax.imshow(colorize(image[channel,::4,::4], color),alpha=0.5)

    channel_labels = create_channel_labels(channels)
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=channel_labels[channel]) 
                    for channel, color in zip(channels, colors)]

    ax.legend(handles=legend_elements, loc='upper right', fontsize='x-small')

    fig.suptitle(f'sample: {sample_name}, overlay of channels: {channels}')
    plt.savefig(output, dpi=image.shape[1]//20)
    plt.close()

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    
    file_or_folder = check_inputs_paths(args)
    logger.info(f"Processing {file_or_folder}")
    
    channels = process_numbers(args.channels)
    logger.info(f"Channels to overlay: {channels}")

    if file_or_folder == "single_file":
        get_overlay(args.input, args.output, channels)
        logger.info(f"Overlay of single file saved to {args.output}")

    elif file_or_folder == "directory":
        files = natsort.natsorted(os.listdir(args.input))
        for file in tqdm(files):
            if file.endswith(".tif"):
                logger.info(f"Processing {file}")
                input_file = os.path.join(args.input, file)
                output_file = os.path.join(args.output, file.split(".")[0] + "nuclear_overlay.png")
                get_overlay(input_file, output_file, channels)
                logger.debug(f"Overlay of {file} saved to {output_file}")
    logger.info("Done")

if __name__ == "__main__":
    main()



"""
Example usage:

python stitchingQC.py \
--input /Users/jnimoca/Jose_BI/data/HN_Segmentation_Tests/HN26_Core10_subset.ome.tif \
--output /Users/jnimoca/Jose_BI/data/HN_Segmentation_Tests/HN26_Core10_subset_nuclear_overlay.png \
--channels "[4, 9, 14]"

python stitchingQC.py \
--input /Users/jnimoca/Jose_BI/data/HN_Segmentation_Tests/ \
--output /Users/jnimoca/Jose_BI/data/HN_Segmentation_Tests/output_folder/ \
--channels "[4, 9, 14, 19, 24, 29]"

python stitchingQC.py \
--input /Volumes/RD_Coscia/Sonjas_Head_and_Neck/P21E05_HN26/dearray/ \
--output /Volumes/RD_Coscia/Sonjas_Head_and_Neck/P21E05_HN26/qc/ashlar_qc/ \
--channels "[4, 9, 14, 19, 24, 29]"


"""