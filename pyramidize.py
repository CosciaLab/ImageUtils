#system
from loguru import logger
import argparse
import sys
from os.path import abspath
import time
#IO
import skimage
import tifffile
import czifile
import ome_types
#processing
import numpy as np
import zarr
from skimage.exposure import equalize_adapthist
try:
    from skimage.util.dtype import _convert as dtype_convert
except ImportError:
    from skimage.util.dtype import convert as dtype_convert

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
    inputs.add_argument("-r", "--input", dest="input", action="store", required=True, help="File path to input image.")
    inputs.add_argument("-o", "--output", dest="output", action="store", required=True, help="Path to output image.")
    inputs.add_argument("-p", "--pixel-size", dest="pixel_size", action="store", type=float, required=False, help="Image pixel size")
    inputs.add_argument("-t", "--tile-size", dest="tile_size", action="store", type=int, default=1072, help="Tile size for pyramid generation (must be divisible by 16)")
    inputs.add_argument("-l", "--log-level", dest="loglevel", default='INFO', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help='Set the log level (default: INFO)')
    arg = parser.parse_args()
    # Standardize paths
    arg.input = abspath(arg.input)
    arg.pixel_size = float(arg.pixel_size)
    return arg

def load_image_into_array(path_to_image):
    """ Load image into numpy array"""
    try:
        if path_to_image.endswith(".czi"):
            logger.info(f"Reading CZI file with shape {czifile.CziFile(path_to_image).shape}")
            time_st = time.time()
            image = np.squeeze(czifile.imread(path_to_image).astype("uint16"))
            logger.info(f"Loaded image in {time.time() - time_st:.1f} seconds")
        else:
            image = tifffile.imread(path_to_image).astype("uint16")
        logger.info(f"Image stored as array with shape: {image.shape}, and dtype: {image.dtype}")
        return image
    except FileNotFoundError:
        logger.error(f"File not found: {path_to_image}")
        sys.exit(1)
    except MemoryError:
        logger.error("Memory error while loading image, probably more RAM needed")
        sys.exit(1)
    except Exception as err:
        logger.error(f"Error loading image: {err}")
        sys.exit(1)

def create_metadata(pixel_size):
    """
    Create metadata dictionary for the image
    Consider expanding to include channel names, etc.
    """
    metadata = {
        "Creator": "Jose Nimo",
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "\u00b5m",
        },
    }
    return metadata

#parameters from image
def set_parameters_from_image(image, tile_size=1072):
    """
    Set parameters for the image pyramid generation
    """
    image_parameters = {}
    image_parameters["dtype"] = image.dtype
    image_parameters["xy_shape"] = image.shape[-2:]
    image_parameters["num_channels"] = image.shape[0]
    image_parameters["num_levels"] = (np.ceil(np.log2(max(1, max(image_parameters["xy_shape"]) / tile_size))) + 1).astype(int)
    image_parameters["factors"] = 2 ** np.arange(image_parameters["num_levels"])
    image_parameters["shapes"] = (np.ceil(np.array(image_parameters["xy_shape"]) / image_parameters["factors"][:, None])).astype(int)
    image_parameters["cxy_shapes_all_levels"] = []
    for shape in image_parameters["shapes"]:
        image_parameters["cxy_shapes_all_levels"].append((image_parameters["num_channels"], shape[0], shape[1]))
    image_parameters["tip_level"] = np.argmax(np.all(image_parameters["shapes"] < tile_size, axis=1))
    image_parameters["tile_shapes"] = [(tile_size, tile_size) if i <= image_parameters["tip_level"] else None for i in range(len(image_parameters["shapes"]))]

    logger.debug(f"Image x,y dimensions: {image_parameters['xy_shape']}")
    logger.debug(f"Number of channels: {image_parameters['num_channels']}")
    logger.debug(f"Number of levels: {image_parameters['num_levels']}")
    logger.info(f"Factors: {image_parameters['factors']}")
    logger.info (f"Pyramid levels: {' '.join(str(sublist) for sublist in image_parameters['shapes'])}")
    logger.debug(f"Shapes for all levels: {image_parameters['cxy_shapes_all_levels']}")
    logger.debug(f"Tip level: {image_parameters['tip_level']}, the pyramid level where all dimensions are smaller than the tile size")
    logger.debug(f"Tile shapes for each level: {[(i,tileshape) for i,tileshape in enumerate(image_parameters['tile_shapes'])]}")

    return image_parameters

def tile_generator_lowres(level, level_full_shapes, tile_shapes, outpath, scale):
    """
    Generate tiles for lower resolution levels
    tiffwriter takes in generator and writes to file

    Parameters:
    Level: int - pyramid level
    level_full_shapes: list - list of tuples with shapes for each level
    tile_shapes: list - list of tuples with tile shapes for each level
    outpath: string - path to the tiff file
    scale: int - scale factor for downscaling
    """
    logger.info(f"Generating tiles for level {level}")
    #info from image
    num_channels, h, w = level_full_shapes[level]
    tileshape = tile_shapes[level] or (h, w)
    #read tiff with level 0 (highest res)
    tiff = tifffile.TiffFile(outpath)
    zarrImage = zarr.open(tiff.aszarr(series=0, level=level - 1, squeeze=False))
    #generate tiles
    for channel in range(num_channels):
        logger.info(f"    processing channel {channel + 1}/{num_channels}")
        # sys.stdout.flush()
        tileheight = tileshape[0] * scale
        tilewidth = tileshape[1] * scale
    
        for y in range(0, zarrImage.shape[1], tileheight):        
            for x in range(0, zarrImage.shape[2], tilewidth):
                a = zarrImage[channel, y : y + tileheight, x : x + tilewidth, 0]
                a = skimage.transform.downscale_local_mean(a, (scale, scale))
                if np.issubdtype(zarrImage.dtype, np.integer):
                    a = np.around(a)
                a = a.astype("uint16")
                yield a

def detect_pixel_size(img_path, pixel_size=None):
    """
    Detect pixel size from metadata
    """
    if pixel_size is None:
        print("Pixel size overwrite not specified")
        try:
            metadata = ome_types.from_tiff(img_path)
            pixel_size = metadata.images[0].pixels.physical_size_x
        except Exception as err:
            print(err)
            print("Pixel size detection using ome-types failed")
            pixel_size = None
    return pixel_size

def main():
    args = get_args()
    # logging setup
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}", level=args.loglevel.upper())
    # process image
    image = load_image_into_array(args.input)
    args.pixel_size = detect_pixel_size(args.input, args.pixel_size)
    metadata = create_metadata(args.pixel_size)
    image_parameters = set_parameters_from_image(image, tile_size=args.tile_size)

    with tifffile.TiffWriter(args.output, ome=True, bigtiff=True) as tiff:
        # save image the highest resolution level
        tiff.write(
            data=image,
            metadata=metadata,
            shape=image_parameters["cxy_shapes_all_levels"][0],
            subifds=int(image_parameters["num_levels"] - 1),
            dtype=image_parameters["dtype"],
            resolution=(10000 / args.pixel_size, 10000 / args.pixel_size),
            resolutionunit='CENTIMETER',
            tile=image_parameters["tile_shapes"][0],
        )
        # create tiles for the rest of the levels
        for level, (shape, tile_shape) in enumerate(zip(image_parameters["cxy_shapes_all_levels"][1:], image_parameters["tile_shapes"][1:]), start=1):
            tiff.write(
                data=tile_generator_lowres(level, image_parameters["cxy_shapes_all_levels"], image_parameters["tile_shapes"], args.output, scale=2),
                shape=shape,
                subfiletype=1,
                dtype=image_parameters["dtype"],
                tile=tile_shape,
            )
    logger.info(f"Pyramid image saved to {args.output}")

if __name__ == "__main__":
    st = time.time()
    main()
    logger.info(f"Execution time: {time.time() - st:.1f} seconds")


""" 
Example usage:

python pyramidize.py \
--log-level "INFO" \
--input "/Jose_BI/resize/Subset_HN0041TMA_1.czi" \
--output "/Jose_BI/resize/test_3.ome.tif" \
--pixel-size 0.354 \
--tile-size 1072 
"""