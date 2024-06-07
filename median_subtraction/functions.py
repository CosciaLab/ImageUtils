import os
if not os.name:
    os.environ['MPLCONFIGDIR'] = '/data/cephfs-1/work/groups/ag-coscia/rafael/scripts'
import tifffile, palom, os.path, quantify, cv2
from skimage.filters import median, gaussian
from skimage.morphology import disk
import dask.array as da
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from cellpose import models, io
import matplotlib.pyplot as plt


# @ray.remote
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
        # segmentation_folder = 'mesmer-' + base_input.replace('.ome.tif', '')
        segmentation_folder = base_input.replace('.ome.tif', '')
        segmentation_path = os.path.join(input_path, 'segmentation', segmentation_folder)
        segmentation = os.path.join(segmentation_path, 'cell.tif')
        segmentation_2 = os.path.join(input_path, 'segmentation', segmentation_folder+ '.tif')

        if not os.path.exists(os.path.join(input_path, 'segmentation')):
            os.makedirs(os.path.join(input_path, 'segmentation'))

        if not os.path.exists(segmentation_path):
            os.makedirs(segmentation_path)

        # Marker list
        markers_path = os.path.join(input_path, 'markers.csv')
    except:
        print('Cannot create paths')
        pass
    try:
        if os.path.exists(output):
            original_image = tifffile.imread(input)
            med_rm_img = tifffile.imread(output)
        else:
            original_image = tifffile.imread(input)
    except:
        print('Cannot open image')
        pass

    try:
        markers_table = pd.read_csv(markers_path)
        markers_table = markers_table[~markers_table['marker_name'].str.contains('_bg')]

        nucleus_channels = markers_table[markers_table['marker_name'].str.contains('DAPI')]
        nucleus_channels = nucleus_channels.Channel_number - 1
        nucleus_img = original_image[nucleus_channels]
        print('Pulled from original_image')

        # Find 75th percentile for max
        nucleus_img = np.quantile(nucleus_img, 0.75, axis=0)
        print('Nuclei stains combined')

        nucleus_img = median_blur_remove(nucleus_img)
        nucleus_img = median(nucleus_img, disk(1.5))
        print('Median removed')

        nucleus_img = nucleus_img - np.quantile(nucleus_img, 0.75)
        nucleus_img[nucleus_img < 0] = 0
        print('Rescaled min')

        nucleus_img = nucleus_img / np.quantile(nucleus_img, 0.99)
        nucleus_img[nucleus_img > 1] = 1
        print('Rescaled max')

        nucleus_img = nucleus_img * 255
        nucleus_img = nucleus_img.astype(np.uint8)
        print('8-bit')

        tifffile.imwrite(os.path.join(segmentation_path, 'nucleus.tif'), nucleus_img, imagej=True, compression='lzw')
    except:
        print('Cannot create nucleus img')

    try:
        cytosol_channels = markers_table[~markers_table['marker_name'].str.contains('DAPI')]
        cytosol_channels = cytosol_channels.Channel_number - 1
        cytosol_img = original_image[cytosol_channels]
        cytosol_img = np.quantile(cytosol_img, 0.75, axis=0)
        cytosol_img = median_blur_remove(cytosol_img)
        cytosol_img = median(cytosol_img, disk(1.5))
        cytosol_img[cytosol_img < 0] = 0
        cytosol_img = maximum_filter(cytosol_img, size=(3, 3))
        cytosol_img = median(cytosol_img, disk(3))
        cytosol_img = cytosol_img - np.quantile(cytosol_img, 0.75)
        cytosol_img[cytosol_img < 0] = 0
        cytosol_img = cytosol_img / np.quantile(cytosol_img, 0.995)
        cytosol_img[cytosol_img > 1] = 1
        cytosol_img = cytosol_img * 255
        cytosol_img = cytosol_img.astype(np.uint8)
        cytosol_img = np.maximum(cytosol_img, nucleus_img)
        tifffile.imwrite(os.path.join(segmentation_path, 'cytosol.tif'), cytosol_img, imagej=True, compression='lzw')
    except:
        print('Cannot create cytosol img')

    try:
        print('Stacked')
        stacked_channels = np.dstack((cytosol_img, nucleus_img))
    except:
        print('Cannot stack cytosol and nucleus')

    try:
        io.logger_setup()
        # model = models.CellposeModel(pretrained_model='/data/cephfs-1/work/groups/ag-coscia/rafael/scripts/img_analysis/cytotorch_3')
        # model = models.CellposeModel(pretrained_model='/Users/rdeliza/Downloads/models/cytotorch_3')
        print('Pulled model')
    except:
        print('Cannot pull model')
    try:
        masks, flows, styles = model.eval(stacked_channels, diameter=38, channels=[1, 2])
        print('Applied model')
    except:
        print('Cannot segment')
    try:
        print('Max filter')
        mask_extended = maximum_filter(masks, size=(15, 15))
        masks[masks == 0] = mask_extended[masks == 0]
        print('Save to ' + segmentation)
        tifffile.imwrite(segmentation, masks, imagej=True, compression='lzw')
        tifffile.imwrite(segmentation_2, masks)
        print('Write segmentation label img')
    except:
        print('Cannot save segmentation')

    try:
        new_cytosol_img = cytosol_img - nucleus_img
        # Convert nucleus_img and cytosol_img to RGB and apply colors
        nucleus_img_colored = np.stack([nucleus_img, np.zeros_like(nucleus_img), nucleus_img], axis=-1)  # Magenta
        cytosol_img_colored = np.stack(
            [np.zeros_like(new_cytosol_img), new_cytosol_img, np.zeros_like(new_cytosol_img)], axis=-1)  # Green
        # Combine nucleus and cytosol images
        combined_img = np.clip(nucleus_img_colored + cytosol_img_colored, 0, 255)
        print('Marker-mask to rgb')

        # Step 3: Create an outline mask by eroding and subtracting
        # Step 3: Create an outline mask by eroding and subtracting
        mask_extended = maximum_filter(masks, size=(3, 3))
        mask_reduced = minimum_filter(masks, size=(6, 6))
        mask_difference = mask_reduced - mask_extended
        mask_difference = mask_difference > 0
        outline_masks = mask_difference.astype(np.uint8)
        print('Outline computed')

        # Step 4: Create an overlay with yellow outlines
        overlay_image = combined_img.copy()
        overlay_image[outline_masks == 1] = [255, 255, 0]  # Yellow
        print('Made yellow')

        # Display the result for verification
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.show()
        print('Plotted')

        # Step 5: Save the resulting image as a PNG file
        output_image_path = os.path.join(segmentation_path, 'overlay.png')
        io.imsave(output_image_path, overlay_image.astype(np.uint8))
        print('Overlay saved')
    except:
        print('Cannot create overlay')
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
        else:
            print('Median-removed image exists')
    except:
        print('Cannot remove median')
        pass

    try:
        cytosol_channels = markers_table[~markers_table['marker_name'].str.contains('DAPI')]
        cytosol_channels = cytosol_channels.Channel_number - 1
        cytosol_img = original_image[cytosol_channels]
        cytosol_img = np.quantile(cytosol_img, 0.75, axis=0)
        cytosol_img = median_blur_remove(cytosol_img)
        cytosol_img = median(cytosol_img, disk(1.5))
        cytosol_img[cytosol_img < 0] = 0
        cytosol_img = maximum_filter(cytosol_img, size=(3, 3))
        cytosol_img = median(cytosol_img, disk(3))
        cytosol_img = cytosol_img - np.quantile(cytosol_img, 0.75)
        cytosol_img[cytosol_img < 0] = 0
        cytosol_img = cytosol_img / np.quantile(cytosol_img, 0.995)
        cytosol_img[cytosol_img > 1] = 1
        cytosol_img = cytosol_img * 255
        cytosol_img = cytosol_img.astype(np.uint8)
        cytosol_img = np.maximum(cytosol_img, nucleus_img)
        tifffile.imwrite(os.path.join(segmentation_path, 'cytosol.tif'), cytosol_img, imagej=True, compression='lzw')
    except:
        print('Cannot create cytosol img')

    try:
        print('Stacked')
        stacked_channels = np.dstack((cytosol_img, nucleus_img))
    except:
        print('Cannot stack cytosol and nucleus')

    try:
        io.logger_setup()
        model = models.CellposeModel(pretrained_model='/data/cephfs-1/work/groups/ag-coscia/rafael/scripts/img_analysis/cytotorch_3')
        # model = models.CellposeModel(pretrained_model='/Users/rdeliza/Scripts/img_analysis/cytotorch_3')
        print('Pulled model')
    except:
        print('Cannot pull model')
    try:
        masks, flows, styles = model.eval(stacked_channels, diameter=38, channels=[1, 2])
        print('Applied model')
    except:
        print('Cannot segment')
    try:
        print('Max filter')
        mask_extended = maximum_filter(masks, size=(15, 15))
        masks[masks == 0] = mask_extended[masks == 0]
        print('Save to '+ segmentation)
        tifffile.imwrite(segmentation, masks, imagej=True, compression='lzw')
        tifffile.imwrite(segmentation_2, masks)
        print('Write segmentation label img')
    except:
        print('Cannot save segmentation')

    try:
        new_cytosol_img = cytosol_img - nucleus_img
        # Convert nucleus_img and cytosol_img to RGB and apply colors
        nucleus_img_colored = np.stack([nucleus_img, np.zeros_like(nucleus_img), nucleus_img], axis=-1)  # Magenta
        cytosol_img_colored = np.stack(
            [np.zeros_like(new_cytosol_img), new_cytosol_img, np.zeros_like(new_cytosol_img)], axis=-1)  # Green
        # Combine nucleus and cytosol images
        combined_img = np.clip(nucleus_img_colored + cytosol_img_colored, 0, 255)
        print('Marker-mask to rgb')

        # Step 3: Create an outline mask by eroding and subtracting
        # Step 3: Create an outline mask by eroding and subtracting
        mask_extended = maximum_filter(masks, size=(3, 3))
        mask_reduced = minimum_filter(masks, size=(6, 6))
        mask_difference = mask_reduced - mask_extended
        mask_difference = mask_difference > 0
        outline_masks = mask_difference.astype(np.uint8)
        print('Outline computed')

        # Step 4: Create an overlay with yellow outlines
        overlay_image = combined_img.copy()
        overlay_image[outline_masks == 1] = [255, 255, 0]  # Yellow
        print('Made yellow')

        # Display the result for verification
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.show()
        print('Plotted')

        # Step 5: Save the resulting image as a PNG file
        output_image_path = os.path.join(segmentation_path, 'overlay.png')
        io.imsave(output_image_path, overlay_image.astype(np.uint8))
        print('Overlay saved')
    except:
        print('Cannot create overlay')
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
        else:
            print('Median-removed image exists')
    except:
        print('Cannot remove median')
        pass

    try:
        # Quantify cells
        print('Measuring cells')
        quant_table = quantify.main(img=med_rm_img, labels_path=segmentation_2, markers_path=markers_path)

        # Export quantification
        table_name = base_input.replace('.ome.tif', '')
        # table_path = os.path.join(input_path, 'quantification', table_name + '.parquet')
        print('Saving cell measurements table')
        #quant_table.to_parquet(table_path)
        #print(table_path)

        csv_path = os.path.join(input_path, 'quantification', table_name + '.csv')
        quant_table.to_csv(csv_path, index=False)  # Set index=False to not include row numbers in the CSV file
        print(csv_path)
    except:
        print('Cannot measure cells')
        pass

    try:
        # Get markers for TIFF metadata
        print('Pulling markers')
        # part 0, load markers
        try:
            markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#', sep=';')
        except:
            pass

        try:
            markers = pd.read_csv(markers_path, dtype={0: 'int16', 1: 'int16', 2: 'str'}, comment='#', sep=',')
        except:
            print('Could not read ' + markers_path)
            pass
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
