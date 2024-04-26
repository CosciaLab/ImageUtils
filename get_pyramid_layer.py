from loguru import logger
import argparse
import sys
from os.path import abspath
import time
from pyometiff import OMETIFFWriter
import tifffile


def get_args():
    """
    Get arguments from command line
    """
    # Script description
    description = """Multiscale ome.tif from numpy array"""
    # Add parser
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sections
    inputs = parser.add_argument_group(title="Required Input", description="Path to required input file")
    inputs.add_argument("-r", "--input",        dest="input",       action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output",       dest="output",      action="store", required=True, help="Path to output image.")
    inputs.add_argument("-l", "--level",        dest="level",       action="store", required=True, type=int, help="Level of pyramid to extract")
    inputs.add_argument("-ll", "--log-level",   dest="loglevel",    default='INFO', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help='Set the log level (default: INFO)')
    inputs.add_argument("-p", "--pixel-size",   dest="pixel_size",  action="store", required=False, type=float,  help="Image pixel size")
    arg = parser.parse_args()
    # Standardize paths
    arg.input = abspath(arg.input)
    arg.output = abspath(arg.output)
    return arg

def check_image_and_levels(image_path, level):
    """
    Check if image and level are compatible
    """
    try:
        with tifffile.TiffFile(image_path) as tif:
            logger.debug(f"Number of series: {len(tif.series)}")
            logger.debug(f"Number of channels: {len(tif.pages)}")
            for i, level in enumerate(tif.series[0].levels):
                logger.debug(f"Level {i}: {level.shape}")
            # assert level < len(tif.series[0].levels), f"Level {level} not found in image"
    except Exception as err:
        logger.error(f"Error reading image: {err}")
        sys.exit(1)

def get_pyramid_layer(image_path, level, pixel_size, output_path):
    """
    Get pyramid layer from image
    """
    # a string describing the dimension ordering
    dimension_order = "CYX"

    logger.info(f"Reading image: {image_path}")
    time_st = time.time()
    with tifffile.TiffFile(image_path) as tif:
        image = tif.series[0].levels[level].asarray()
    logger.info(f"Read image layer in {time.time() - time_st:.1f} seconds")
    logger.debug(f"Image shape: {image.shape}")

    # metadata 
    downscale_pixel_size = pixel_size * 2**level
    logger.info(f"Original pixel size: {pixel_size:.2f}/µm")
    logger.info(f"Downscaled pixel size: {downscale_pixel_size:.2f}/µm")

    metadata = {
        "PhysicalSizeX" : downscale_pixel_size,
        "PhysicalSizeXUnit" : "µm",
        "PhysicalSizeY" : downscale_pixel_size,
        "PhysicalSizeYUnit" : "µm",
    }

    writer = OMETIFFWriter(
        fpath=output_path,
        dimension_order=dimension_order,
        array=image,
        metadata=metadata)
    
    writer.write()

def main():
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    check_image_and_levels(args.input, args.level)
    get_pyramid_layer(args.input, args.level, args.pixel_size, args.output)

if __name__ == "__main__":
    st = time.time()
    main()
    logger.info(f"Execution time: {time.time() - st:.1f} seconds")