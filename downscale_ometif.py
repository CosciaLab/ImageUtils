import tifffile
from skimage.transform import rescale
from ome_types import from_tiff, to_xml
import numpy as np

def get_metadata(img_path, downscale_factor, new_file_name):
    """
    Detect image dimensions and channel information from metadata,
    adjust them based on the downscale factor, and update the file_name for the new image.
    """
    ome_metadata = from_tiff(img_path)
    pixels = ome_metadata.images[0].pixels

    # adjust the pixel dimensions (SizeX, SizeY) based on the downscale factor
    pixels.size_x = int(pixels.size_x * downscale_factor)
    pixels.size_y = int(pixels.size_y * downscale_factor)

    # update the file_name in the tiff_data_blocks, if it exists
    # if the file_name under tiff_data_blocks is not updated, the new image is not read properly
    for tiff_data_block in pixels.tiff_data_blocks:
        if tiff_data_block.uuid is not None:
            tiff_data_block.uuid.file_name = new_file_name

    return ome_metadata

def downscale_ome_tiff(image_path, downscale_factor, output_path):
    # open the image and extract metadata
    with tifffile.TiffFile(image_path) as tif:
        img = tif.asarray()

    # downscale the image
    img_resized = rescale(img, (1, downscale_factor, downscale_factor), anti_aliasing=True, preserve_range=True).astype(np.uint16)

    # FIXME: The following code is not needed for now, but may be needed in the future.
    #        Rescale currently works as intended (images look the same), but the maximum values differ
    #        between the original and downscaled images. This may be due to the rescaling method used.  
    '''
    # Normalize and scale the image to the range of uint16
    img_min = img_resized.min()
    img_max = img_resized.max()
    max_value = np.iinfo(np.uint16).max
    img_resized = ((img_resized - img_min) / (img_max - img_min) * max_value).astype(np.uint16)
    '''

    # create metadata with adjusted dimensions (SizeX, SizeY) and correct file name
    new_file_name = output_path.split("\\")[-1]  # Extract just the file name from the path
    ome_metadata = get_metadata(image_path, downscale_factor, new_file_name)
    
    # Convert OME metadata to XML
    ome_xml = to_xml(ome_metadata)
    
    # encode the OME-XML string as UTF-8 bytes
    ome_xml_encoded = ome_xml.encode('utf-8')
    
    # save the downscaled image as a new OME-TIFF file
    with tifffile.TiffWriter(output_path) as tif:
        tif.write(img_resized, description=ome_xml_encoded)

# usage
downscale_factor = 0.5
input_path = r'C:\Users\mtrinh\Documents\small_subset.ome.tif'
output_path = r'C:\Users\mtrinh\Documents\small_subset_downscaled.ome.tif'

downscale_ome_tiff(input_path, downscale_factor, output_path)
