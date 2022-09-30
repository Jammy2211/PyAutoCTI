import numpy as np

import autoarray as aa
from autoarray import Array1D, Region1D
from autoarray.dataset import abstract_dataset
from autocti import exc
from autocti.dataset_1d.dataset_1d.settings import SettingsDataset1D
from autocti.layout.one_d import Layout1D


class Dataset1D(abstract_dataset.AbstractDataset):
    def __init__(
            self,
            data: aa.Array1D,
            noise_map: aa.Array1D,
            pre_cti_data: aa.Array1D,
            layout: Layout1D,
            settings: SettingsDataset1D = SettingsDataset1D(),
    ):

        super().__init__(data=data, noise_map=noise_map, settings=settings)

        self.data = data
        self.noise_map = noise_map
        self.pre_cti_data = pre_cti_data
        self.layout = layout

    def apply_mask(self, mask: aa.Mask1D) -> "Dataset1D":

        data = aa.Array1D.manual_mask(array=self.data, mask=mask).native
        noise_map = aa.Array1D.manual_mask(
            array=self.noise_map.astype("float"), mask=mask
        ).native

        return Dataset1D(
            data=data,
            noise_map=noise_map,
            pre_cti_data=self.pre_cti_data,
            layout=self.layout,
        )

    def apply_settings(self, settings: SettingsDataset1D) -> "Dataset1D":

        return self

    @classmethod
    def from_fits(
            cls,
            layout,
            data_path,
            pixel_scales,
            data_hdu=0,
            noise_map_path=None,
            noise_map_hdu=0,
            noise_map_from_single_value=None,
            pre_cti_data_path=None,
            pre_cti_data_hdu=0,
            pre_cti_data=None,
    ):

        data = aa.Array1D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )

        if noise_map_path is not None:
            noise_map = aa.util.array_1d.numpy_array_1d_via_fits_from(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            noise_map = np.ones(data.shape_native) * noise_map_from_single_value

        noise_map = aa.Array1D.manual_native(array=noise_map, pixel_scales=pixel_scales)

        if pre_cti_data_path is not None and pre_cti_data is None:
            pre_cti_data = aa.Array1D.from_fits(
                file_path=pre_cti_data_path,
                hdu=pre_cti_data_hdu,
                pixel_scales=pixel_scales,
            )
        else:
            raise exc.LayoutException(
                "Cannot estimate pre_cti_data data from non-uniform charge injectiono pattern"
            )

        pre_cti_data = aa.Array1D.manual_native(
            array=pre_cti_data.native, pixel_scales=pixel_scales
        )

        return Dataset1D(
            data=data, noise_map=noise_map, pre_cti_data=pre_cti_data, layout=layout
        )

    def output_to_fits(
            self, data_path, noise_map_path=None, pre_cti_data_path=None, overwrite=False
    ):

        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)
        self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
        self.pre_cti_data.output_to_fits(
            file_path=pre_cti_data_path, overwrite=overwrite
        )

    @classmethod
    def from_pixel_line_dict(
            cls,
            pixel_line_dict: dict,
            size: int,
    ) -> "Dataset1D":
        serial_distance, _ = pixel_line_dict["location"]

        def make_array(data):
            array = np.zeros(size)
            array[serial_distance: serial_distance + len(data)] = data
            return Array1D.manual_slim(array, pixel_scales=0.1)

        return Dataset1D(
            data=make_array(pixel_line_dict["data"]),
            noise_map=make_array(pixel_line_dict["noise"]),
            pre_cti_data=make_array(np.array([pixel_line_dict["flux"]])),
            layout=Layout1D(
                shape_1d=(size,),
                region_list=[Region1D(
                    region=(serial_distance, serial_distance + 1)
                )]
            ),
        )
