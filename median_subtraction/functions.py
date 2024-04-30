import tifffile, palom, os, os.path, quantify, cv2
# import ray, time
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
import dask.array as da
import pandas as pd
import pyarrow, fastparquet


#@ray.remote
def median_blur_remove(img):
    try:
        # Scale down 8x
        # Determine the new dimensions (1/8th of original size)
        old_width = img.shape[1]
        old_height = img.shape[0]
        new_width = old_width // 8
        new_height = old_height // 8

        # Blur it before down sampling
        med_blur_img = median(img, disk(3.5))
        # Resize the median filtered image to the new dimensions
        small_img = cv2.resize(med_blur_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # Blur image
        med_blur_img = median(small_img, disk(50.5))
        # Scale back up
        med_blur_img = cv2.resize(med_blur_img, (old_width, old_height))
        # Remove median
        med_rm = img - np.minimum(med_blur_img, img)

        return med_rm
    except:
        print('median_blur_remove error')
        pass

def median_data(input):
    try:
        # Get all directories path
        input_path = os.path.dirname(input)
        input_path = os.path.dirname(input_path)

        # Get image name
        base_input = os.path.basename(input)

        # Make output path
        output = os.path.join(input_path, 'median_subtracted', base_input)

        # Make segmentation path
        segmentation_folder = 'mesmer-' + base_input.replace('.ome.tif', '')
        segmentation = os.path.join(input_path, 'segmentation', segmentation_folder, 'cell.tif')

        # Marker list
        markers_path = os.path.join(input_path, 'markers.csv')
    except:
        print('Cannot create paths')
        pass
    try:
        if os.path.exists(output):
            med_rm_img = tifffile.imread(output)
        else:
            original_image = tifffile.imread(input)
    except:
        print('Cannot open image')
        pass

    try:
        if not os.path.exists(output):
            # Get channel count
            channels = original_image.shape[0]
            channels = range(0, channels)
            # '''
            stack = []
            for channel in channels:
                print('Channel ' + str(channel))
                channel_image = original_image[channel, :, :]
                blurred_channel_image = median_blur_remove(channel_image)
                stack.append(blurred_channel_image)

            # Convert the list of images into a numpy array to represent the stack
            med_rm_img = np.array(stack)
            '''
            # Run in parallel
            # Get the number of CPU cores
            num_cpus = os.cpu_count()
            num_cpus = num_cpus//3
            # Get the total amount of RAM in bytes
            total_ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

            # Convert bytes to gigabytes
            total_ram_gb = total_ram_bytes / (1024 ** 3)
            total_ram_gb = total_ram_gb // 10

            use_n_cpu = min(total_ram_gb, num_cpus)
            use_n_cpu = use_n_cpu.__int__()

            ray.shutdown()
            ray.init(num_cpus=use_n_cpu)
            result_ids = []
            for channel in channels:
                print('Channel ' + channel.__str__())
                channel_image = original_image[channel, :, :]
                result_id = median_blur_remove.remote(channel_image)
                result_ids.append(result_id)
            print('Append channels')
            med_rm_img = ray.get(result_ids)
            print('Append finished')
            ray.shutdown()
            time.sleep(10)
            ray.shutdown()
            print('Ray shutdown')
            '''
        else:
            print('Median-removed image exists')
    except:
        print('Cannot remove median')
        pass

    try:
        # Quantify cells
        print('Measuring cells')
        quant_table = quantify.main(img=med_rm_img, labels_path=segmentation, markers_path=markers_path)

        # Export quantification
        table_name = base_input.replace('.ome.tif', '')
        table_path = os.path.join(input_path, 'quantification', table_name + '.parquet')
        print('Saving cell measurements table')
        quant_table.to_parquet(table_path)
        print(table_path)
    except:
        print('Cannot measure cells')
        pass

    try:
        # Get markers for TIFF metadata
        print('Pulling markers')
        markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#', sep=';')
        markers = markers['marker_name'].astype(str).tolist()
    except:
        print('Cannot open marker names')
        pass

    try:
        if not os.path.exists(output):
            # Save
            print('Saving median-subtracted image')
            da_img = da.from_array(med_rm_img, chunks=(1000, 1000, med_rm_img.__len__()))
            palom.pyramid.write_pyramid(da_img, output, compression='lzw', channel_names=markers, downscale_factor=2,
                                        pixel_size=0.3457)

    except:
        print('Cannot save image')
        pass
