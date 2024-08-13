import tifffile
from skimage.transform import rescale
from ome_types import from_tiff

def get_metadata(img_path):
    """
    Detect pixel size from metadata and save it for the new image
    """
    ome_metadata = from_tiff(img_path)
    pixel_size = ome_metadata.images[0].pixels.physical_size_x

    metadata = {
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "\u00b5m",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "\u00b5m",
        },
    }
    return metadata

def downscale_ome_tiff(image_path, channel, downscale_factor, output_path):
    # Open the image and extract metadata
    with tifffile.TiffFile(image_path) as tif:
        img = tif.asarray()

    # Extract the specified channel
    image = img[channel] 
    
    # Downscale the specified channel
    img_resized = rescale(image, downscale_factor, mode='reflect', preserve_range=True, channel_axis=None)
    img_resized = img_resized.astype(image.dtype)
    
    # Create metadata 
    metadata_new = get_metadata(image_path)
    
    # Save the downscaled channel as a new OME-TIFF file
    with tifffile.TiffWriter(output_path) as tif:
        tif.write(img_resized, metadata=metadata_new)

# Example usage
downscale_factor = 0.3005
channel = 8

# Save the downscaled image to thse same directory and add "_downscaled" to the filename
input_path = r"C:\Users\mtrinh\Documents\coding\master-thesis\sample_data\LungCancer\P09_E02_Lung_Glass.ome.tif"
output_path = input_path.replace(".ome.tif", f"_downscaled_ch{channel}_df{downscale_factor}.ome.tif")

downscale_ome_tiff(input_path, channel, downscale_factor, output_path)
