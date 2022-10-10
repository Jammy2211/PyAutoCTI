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
        """
        Parse a pixel line output from the warm-pixels script.

        Pixel lines are individual or averaged lines found by searching for
        warm pixels or consistent warm pixels in CCD data. The warm pixel
        and its are extracted and saved as a JSON which can then be loaded
        and fit as part of autocti.

        Parameters
        ----------
        pixel_line_dict
            A dictionary describing a pixel line collection.

            e.g.
            {
                'data': [1., 4., 5.],
                'flux': 150,
                'location': [26, 25],
                'noise': [2., 3., 4.],
                'trail': {
                    'data': [4.0],
                    'flux': 4.0,
                    'noise': [4.47213595499958]
                }
            }

            location
                The location of the warm pixel in (row, column) where row is the
                distance to the serial register - 1
            flux
                The computed flux of the warm pixel prior to CTI
            data
                A 1D array in the parallel direction including a warm pixel
            noise
                The noise map for the pixel line.
            trail
                Includes data, flux and noise for just the extracted pixel line.
                This is a 1D array where the first entry is the warm pixel (FPR)
                and the remaining entries are the trail (EPER)
        size
            The size of the CCD. That is, the number of pixels in the parallel
            direction.

        Returns
        -------
            A Dataset1D initialised to represent the pixel line. The pixel line
            and noise are embedded in Array1Ds of the same size as the array in
            the parallel direction
        """
        serial_distance, _ = map(int, pixel_line_dict["location"])

        def make_array(data):
            array = np.zeros(size)
            array[serial_distance: serial_distance + len(data)] = data
            return Array1D.manual_slim(array, pixel_scales=0.1)

        trail = pixel_line_dict["trail"]

        return Dataset1D(
            data=make_array(trail["data"]),
            noise_map=make_array(trail["noise"]),
            pre_cti_data=make_array(np.array([trail["flux"]])),
            layout=Layout1D(
                shape_1d=(size,),
                region_list=[Region1D(
                    region=(serial_distance, serial_distance + 1)
                )]
            ),
        )
