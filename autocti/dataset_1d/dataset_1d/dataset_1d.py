import numpy as np
from typing import Optional, Dict

import autoarray as aa

from autocti import exc
from autocti.dataset_1d.dataset_1d.settings import SettingsDataset1D
from autocti.extract.settings import SettingsExtract
from autocti.layout.one_d import Layout1D


class Dataset1D(aa.AbstractDataset):
    def __init__(
        self,
        data: aa.Array1D,
        noise_map: aa.Array1D,
        pre_cti_data: aa.Array1D,
        layout: Layout1D,
        fpr_value: Optional[float] = None,
        settings: SettingsDataset1D = SettingsDataset1D(),
        settings_dict: Optional[Dict] = None,
    ):
        super().__init__(data=data, noise_map=noise_map, settings=settings)

        self.data = data
        self.noise_map = noise_map
        self.pre_cti_data = pre_cti_data
        self.layout = layout

        if fpr_value is None:
            fpr_value = np.round(
                np.median(
                    self.layout.extract.fpr.stacked_array_1d_from(
                        array=self.data,
                        settings=SettingsExtract(
                            pixels_from_end=min(
                                5, self.layout.extract.fpr.total_pixels_min
                            )
                        ),
                    )
                ),
                2,
            )

        self.fpr_value = fpr_value

        self.settings_dict = settings_dict

    def apply_mask(self, mask: aa.Mask1D) -> "Dataset1D":
        data = aa.Array1D(values=self.data, mask=mask).native
        noise_map = aa.Array1D(values=self.noise_map.astype("float"), mask=mask).native

        return Dataset1D(
            data=data,
            noise_map=noise_map,
            pre_cti_data=self.pre_cti_data,
            layout=self.layout,
            fpr_value=self.fpr_value,
            settings_dict=self.settings_dict,
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
        settings_dict: Optional[Dict] = None,
    ):
        data = aa.Array1D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )

        if noise_map_path is not None:
            noise_map = aa.util.array_1d.numpy_array_1d_via_fits_from(
                file_path=noise_map_path, hdu=noise_map_hdu
            ).astype("float")
        else:
            noise_map = np.ones(data.shape_native) * noise_map_from_single_value

        noise_map = aa.Array1D.no_mask(values=noise_map, pixel_scales=pixel_scales)

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

        pre_cti_data = aa.Array1D.no_mask(
            values=pre_cti_data.native, pixel_scales=pixel_scales
        )

        return Dataset1D(
            data=data,
            noise_map=noise_map,
            pre_cti_data=pre_cti_data,
            layout=layout,
            settings_dict=settings_dict,
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
                "location": [
                    2,
                    4,
                ],
                "flux": 1234.,
                "data": [
                    5.0,
                    3.0,
                    2.0,
                    1.0,
                ],
                "noise": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            }

            location
                The location of the warm pixel in (row, column) where row is the
                distance to the serial register - 1
            flux
                The computed flux of the warm pixel prior to CTI
            data
                The extracted pixel line. A 1D array where the first entry is
                the warm pixel (FPR) and the remaining entries are the trail
                (EPER)
            noise
                The noise map for the pixel line.
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
            array[serial_distance : serial_distance + len(data)] = data
            return aa.Array1D.no_mask(array, pixel_scales=0.1)

        return Dataset1D(
            data=make_array(pixel_line_dict["data"]),
            noise_map=make_array(pixel_line_dict["noise"]),
            pre_cti_data=make_array(np.array([pixel_line_dict["flux"]])),
            layout=Layout1D(
                shape_1d=(size,),
                region_list=[
                    aa.Region1D(region=(serial_distance, serial_distance + 1))
                ],
            ),
        )
