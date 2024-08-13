import os
import time
import logging
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
from shapely.geometry import shape, mapping
import json
from shapely.affinity import scale
import math

# TODO: use .json as input for file paths and downscale factor
# TODO: make GUI for downscaling for future purposes
# TODO: improve logging

# short manual:
#   1. paste paths to GeoJSON and tiff files
#   2. choose downscaling factor 
#   3. execute code

# Paths
geoj_path = r"D:\Proteomics\Minh\LungCancer\raw_data\LC0004WT_HE_annotations.geojson"
# tiff_path = r"D:\Proteomics\Minh\LungCancer\raw_data\sonja.ome.tif"
tiff_path = r"C:\Users\mtrinh\Documents\LungCancer\P09_E02_Lung_Glass.ome.tif"

# image is two dimensional, therefore we have to scale the GeoJSON by the squareroot of the downscale factor
downscale_factor = 8
dsf_gj = math.sqrt(downscale_factor)

# Configure logging
logging.basicConfig(filename='transformation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    start_time = time.time()

    # Logging entry for initiating downscaling process
    with rasterio.open(tiff_path) as src:
        original_width = src.width
        original_height = src.height
        original_file_size = os.path.getsize(tiff_path) / (1024**3)
        formatted_file_size = formatted_file_size = "{:.2f} GB".format(original_file_size)
        logging.info(f"Initiating downscaling for TIFF File. Input Size: {original_width} x {original_height}, File Size: {formatted_file_size}")

    # Step 1: Read the GeoTIFF file
    with rasterio.open(tiff_path) as src:
        profile = src.profile
        transform = src.transform

        # Step 2: Downscale the GeoTIFF by 8 fold
        new_height = src.height // downscale_factor
        new_width = src.width // downscale_factor
        new_transform = Affine(transform.a * downscale_factor, transform.b, transform.c,
                               transform.d, transform.e * downscale_factor, transform.f)

        # Read all bands and downscale
        downscaled_data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )

        # Save the downscaled GeoTIFF in the transformed folder
        base_dir = os.path.dirname(os.path.dirname(tiff_path))  # Get parent directory of raw_data
        transformed_folder = os.path.join(base_dir, 'transformed')
        os.makedirs(transformed_folder, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(tiff_path))[0] + '_downscaled.tif'
        downscaled_tiff_path = os.path.join(transformed_folder, file_name)
        profile.update(
            dtype=downscaled_data.dtype,
            count=src.count,
            width=new_width,
            height=new_height,
            transform=new_transform
        )
        with rasterio.open(downscaled_tiff_path, 'w', **profile) as dst:
            dst.write(downscaled_data)

        downscaled_file_size = os.path.getsize(downscaled_tiff_path) / (1024 ** 3)
        formatted_downscaled_file_size = "{:.2f} GB".format(downscaled_file_size)
        percentage_reduced = (downscaled_file_size/original_file_size)*100
        formatted_percentage_reduced = "{:.2f} %".format(percentage_reduced)
        # Log information
        logging.info(f"Downscaling of TIFF completed. Output size: {new_width} x {new_height}, File Size: {formatted_downscaled_file_size}")
        logging.info(f"Scaled original TIFF file down to {formatted_percentage_reduced} of the original size")

    # # Step 3: Read the GeoJSON file
    # with open(geoj_path) as f:
    #     geojson_data = json.load(f)

    # # Step 4: Adjust the polygon coordinates and sizes
    # for feature in geojson_data['features']:
    #     geom = shape(feature['geometry'])
    #     # Scale down the polygon coordinates using the affinity.scale method
    #     scaled_geom = scale(geom, xfact=1/dsf_gj, yfact=1/dsf_gj, origin=(0, 0))
    #     # Update the coordinates of the polygon
    #     new_coords = [(x / dsf_gj, y / dsf_gj) for x, y in scaled_geom.exterior.coords]
    #     # Update the geometry in the GeoJSON
    #     feature['geometry'] = mapping(scaled_geom.__class__(new_coords))

    # # Step 5: Write the adjusted polygons to a new GeoJSON file
    # adjusted_geojson_path = os.path.join(transformed_folder, 'downscaled_annotations.geojson')
    # with open(adjusted_geojson_path, 'w') as f:
    #     json.dump(geojson_data, f)

    # Log information
    logging.info("Downscaling of GeoJSON completed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Downscaling Process completed in {elapsed_time:.2f} seconds.")

except Exception as e:
    logging.error(f"An error occurred: {e}")