import spatialdata
import xarray
import numpy as np
from spatialdata.transformations import Identity

def quantiles_nuclear_membrane(sdata, key:str, 
                            nuclear_channels:list=None, 
                            nuclear_quantile:float=0.9,
                            nuclear_min_max_quantiles:list=[0.5, 0.995],
                            membrane_channels:list=None,
                            membrane_quantile:float=0.9,
                            membrane_min_max_quantiles:list=[0.5, 0.995],
                            name:str=None):

    def calculate_quantile(sdata, key: str, channels: list, quantile_value: float = 0.9, return_ArrayData: bool = False, min_max_scaling_quantiles=[0.5, 0.995]):

        image = sdata.images[key]['scale0'].image
        assert type(image) == xarray.core.dataarray.DataArray, 'Image is not a xarray DataArray'
        selected_channels_image = image.sel(c=channels)

        def scaling(data_array, min_max_quantiles=[0.5, 0.995]):
            data_array = data_array.chunk(dict(x=-1, y=-1))
            q_min = data_array.quantile(min_max_quantiles[0], dim=('x', 'y')).values
            q_max = data_array.quantile(min_max_quantiles[1], dim=('x', 'y')).values
            clipped_channel = data_array.clip(min=q_min, max=q_max)
            rescaled_channel = (clipped_channel - q_min) / (q_max - q_min) * 255.0
            rescaled_channel_8bit = rescaled_channel.astype(np.uint8)
            return rescaled_channel_8bit

        data_arrays = [selected_channels_image.sel(c=c) for c in channels]
        scaled_data_arrays = [scaling(data_array, min_max_scaling_quantiles) for data_array in data_arrays]
        concat_data_array = xarray.concat(scaled_data_arrays, dim='c').chunk({'c': -1})
        quantile_projection = concat_data_array.quantile(quantile_value, dim='c').astype(np.uint8)
        quantile_projection_expanded = quantile_projection.expand_dims(dim='c', axis=0)

        if return_ArrayData:
            return quantile_projection_expanded
        else:
            sdata.images[f'{key}_q{int(quantile_value*100)}_projection'] = spatialdata.models.Image2DModel.parse(data=quantile_projection_expanded)
            return sdata

    nuclear_ArrayData = calculate_quantile(sdata, key, nuclear_channels, nuclear_quantile, return_ArrayData=True, min_max_scaling_quantiles=nuclear_min_max_quantiles)
    membrane_ArrayData = calculate_quantile(sdata, key, membrane_channels, membrane_quantile, return_ArrayData=True, min_max_scaling_quantiles=membrane_min_max_quantiles)

    concatenated_array = spatialdata.models.Image2DModel.parse(
        xarray.concat([nuclear_ArrayData, membrane_ArrayData], dim='c'), transformations={"pixels":Identity()}
        )
    if name is None:
        name = f'{key}_n{int(nuclear_quantile*100)}_m{int(membrane_quantile*100)}_mmm{str(membrane_min_max_quantiles[0]).replace(".","")}_proj'

    sdata.images[name] = concatenated_array
    sdata.images[name] = sdata.images[name].assign_coords(c=['Nuclei', 'Membranes'])
    return sdata


def nuclear_quantiles_membrane(sdata, key:str, 
                            nuclear_channel:list=None, 
                            membrane_channels:list=None,
                            membrane_quantile:float=0.9,
                            membrane_min_max_quantiles:list=[0.5, 0.995],
                            name:str=None):

    def calculate_quantile(sdata, key: str, channels: list, quantile_value: float = 0.9, return_ArrayData: bool = False, min_max_scaling_quantiles=[0.5, 0.995]):

        image = sdata.images[key]['scale0'].image
        assert type(image) == xarray.core.dataarray.DataArray, 'Image is not a xarray DataArray'
        selected_channels_image = image.sel(c=channels)

        def scaling(data_array, min_max_quantiles=[0.5, 0.995]):
            data_array = data_array.chunk(dict(x=-1, y=-1))
            q_min = data_array.quantile(min_max_quantiles[0], dim=('x', 'y')).values
            q_max = data_array.quantile(min_max_quantiles[1], dim=('x', 'y')).values
            clipped_channel = data_array.clip(min=q_min, max=q_max)
            rescaled_channel = (clipped_channel - q_min) / (q_max - q_min) * 255.0
            rescaled_channel_8bit = rescaled_channel.astype(np.uint8)
            return rescaled_channel_8bit

        data_arrays = [selected_channels_image.sel(c=c) for c in channels]
        scaled_data_arrays = [scaling(data_array, min_max_scaling_quantiles) for data_array in data_arrays]
        concat_data_array = xarray.concat(scaled_data_arrays, dim='c').chunk({'c': -1})
        quantile_projection = concat_data_array.quantile(quantile_value, dim='c').astype(np.uint8)
        quantile_projection_expanded = quantile_projection.expand_dims(dim='c', axis=0)

        if return_ArrayData:
            return quantile_projection_expanded
        else:
            sdata.images[f'{key}_q{int(quantile_value*100)}_projection'] = spatialdata.models.Image2DModel.parse(data=quantile_projection_expanded)
            return sdata

    nuclear_ArrayData = sdata.images[key]['scale0'].image.sel(c=nuclear_channel).astype(np.uint8)
    # nuclear_ArrayData = nuclear_ArrayData.expand_dims(dim='c', axis=0)
    print(f"nuclear_ArrayData.shape {nuclear_ArrayData.shape}")
    print(f"nuclear_ArrayData.dtype {nuclear_ArrayData.dtype}")
    print(f"type(nuclear_ArrayData) {type(nuclear_ArrayData)}")

    membrane_ArrayData = calculate_quantile(sdata, key, membrane_channels, membrane_quantile, return_ArrayData=True, min_max_scaling_quantiles=membrane_min_max_quantiles)
    print(f"membrane_ArrayData.shape {membrane_ArrayData.shape}")
    print(f"membrane_ArrayData.dtype {membrane_ArrayData.dtype}")
    print(f"type(membrane_ArrayData) {type(membrane_ArrayData)}")

    concatenated_array = spatialdata.models.Image2DModel.parse(
        xarray.concat([nuclear_ArrayData, membrane_ArrayData], dim='c'), transformations={"pixels":Identity()}
        )
    if name is None:
        name = f'{key}_n{int(nuclear_quantile*100)}_m{int(membrane_quantile*100)}_mmm{str(membrane_min_max_quantiles[0]).replace(".","")}_proj'

    sdata.images[name] = concatenated_array
    sdata.images[name] = sdata.images[name].assign_coords(c=['Nuclei', 'Membranes'])
    return sdata