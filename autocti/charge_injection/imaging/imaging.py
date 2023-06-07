import numpy as np
from typing import Optional, List, Dict

import autoarray as aa

from autocti.charge_injection.imaging.settings import SettingsImagingCI
from autocti.charge_injection.layout import Layout2DCI
from autocti.extract.settings import SettingsExtract
from autocti.mask import mask_2d
from autocti import exc


class ImagingCI(aa.Imaging):
    def __init__(
        self,
        data: aa.Array2D,
        noise_map: aa.Array2D,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        noise_scaling_map_dict: Optional[Dict] = None,
        fpr_value: Optional[float] = None,
        settings_dict: Optional[Dict] = None,
    ):
        super().__init__(data=data, noise_map=noise_map)

        self.data = self.image.native
        self.noise_map = self.noise_map.native
        self.pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            cosmic_ray_map = cosmic_ray_map.native

        self.cosmic_ray_map = cosmic_ray_map

        if noise_scaling_map_dict is not None:
            noise_scaling_map_dict = {
                key: noise_scaling_map.native
                for key, noise_scaling_map in noise_scaling_map_dict.items()
            }

        self.noise_scaling_map_dict = noise_scaling_map_dict

        self.layout = layout

        if fpr_value is None:
            fpr_value = np.round(
                np.mean(
                    self.layout.extract.parallel_fpr.median_list_from(
                        array=self.data,
                        settings=SettingsExtract(
                            pixels_from_end=min(
                                10, self.layout.smallest_parallel_rows_within_ci_regions
                            )
                        ),
                    )
                ),
                2,
            )

        self.fpr_value = fpr_value
        self.settings_dict = settings_dict

    @property
    def mask(self):
        return self.image.mask

    @property
    def region_list(self):
        return self.layout.region_list

    @property
    def norm_columns_list(self) -> List:
        """
        The `layout` describes the 2D regions on the data containing charge whose input signal properties are know
        beforehand (e.g. charge injection imaging).

        However, the exact values may not be known and therefore need to be estimated from the image.

        This function estimates the normalization of every column of data in the 2D regions, by taking the median
        of each column. If a mask is applied (e.g. to remove cosmic rays) these pixels are omitted from the median.

        Returns
        -------
        A list of the normalization of every column of the charge regions
        """
        masked_image = np.ma.array(data=self.image, mask=self.image.mask)

        return [
            np.ma.median(masked_image[region.y0 : region.y1, column_index])
            for region in self.region_list
            for column_index in range(region.x0, region.x1)
        ]

    @property
    def pre_cti_data_residual_map(self) -> aa.Array2D:
        """
        The residuals of the data and the pre CTI data.

        This is used to assess whether the pre CTI data has been estimated accurately (e.g. from the FPR of the
        data) and includes e specific set of visualization functions.

        Returns
        -------
        The residual map of the data and pre CTI data.
        """
        return self.data - self.pre_cti_data

    def apply_mask(self, mask: mask_2d.Mask2D) -> "ImagingCI":
        image = aa.Array2D(values=self.image.native, mask=mask)
        noise_map = aa.Array2D(values=self.noise_map.native, mask=mask)

        if self.cosmic_ray_map is not None:
            cosmic_ray_map = aa.Array2D(values=self.cosmic_ray_map.native, mask=mask)

        else:
            cosmic_ray_map = None

        if self.noise_scaling_map_dict is not None:
            noise_scaling_map_dict = {
                key: aa.Array2D(values=noise_scaling_map.native, mask=mask)
                for key, noise_scaling_map in self.noise_scaling_map_dict.items()
            }

        else:
            noise_scaling_map_dict = None

        return ImagingCI(
            data=image,
            noise_map=noise_map,
            pre_cti_data=self.pre_cti_data.native,
            layout=self.layout,
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_dict=noise_scaling_map_dict,
            fpr_value=self.fpr_value,
            settings_dict=self.settings_dict,
        )

    def apply_settings(self, settings: SettingsImagingCI):
        if settings.parallel_pixels is not None:
            dataset = self.layout.extract.parallel_calibration.imaging_ci_from(
                dataset=self, columns=settings.parallel_pixels
            )

            mask = self.layout.extract.parallel_calibration.mask_2d_from(
                mask=self.mask, columns=settings.parallel_pixels
            )

        elif settings.serial_pixels is not None:
            dataset = self.layout.extract.serial_calibration.imaging_ci_from(
                dataset=self, rows=settings.serial_pixels
            )

            mask = self.layout.extract.serial_calibration.mask_2d_from(
                mask=self.mask, rows=settings.serial_pixels
            )

        else:
            return self

        dataset = dataset.apply_mask(mask=mask)

        return dataset

    def set_noise_scaling_map_dict(self, noise_scaling_map_dict: Dict):
        self.noise_scaling_map_dict = {
            key: noise_scaling_map.native
            for key, noise_scaling_map in noise_scaling_map_dict.items()
        }

    @classmethod
    def from_fits(
        cls,
        layout,
        pixel_scales,
        data_path=None,
        image=None,
        data_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_single_value=None,
        pre_cti_data_path=None,
        pre_cti_data_hdu=0,
        pre_cti_data=None,
        cosmic_ray_map_path=None,
        cosmic_ray_map_hdu=0,
        settings_dict: Optional[Dict] = None,
    ) -> "ImagingCI":
        if data_path is not None and image is None:
            ci_image = aa.Array2D.from_fits(
                file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
            )

        elif image is not None:
            ci_image = image

        if noise_map_path is not None:
            ci_noise_map = aa.util.array_2d.numpy_array_2d_via_fits_from(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_native) * noise_map_from_single_value

        ci_noise_map = aa.Array2D.no_mask(
            values=ci_noise_map, pixel_scales=pixel_scales
        )

        if pre_cti_data_path is not None and pre_cti_data is None:
            pre_cti_data = aa.Array2D.from_fits(
                file_path=pre_cti_data_path,
                hdu=pre_cti_data_hdu,
                pixel_scales=pixel_scales,
            )
        elif pre_cti_data is None:
            raise exc.ImagingCIException(
                "Cannot load pre_cti_data from .fits and pass explicit pre_cti_data."
            )

        pre_cti_data = aa.Array2D.no_mask(
            values=pre_cti_data.native, pixel_scales=pixel_scales
        )

        if cosmic_ray_map_path is not None:
            cosmic_ray_map = aa.Array2D.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                pixel_scales=pixel_scales,
            )

        else:
            cosmic_ray_map = None

        return ImagingCI(
            data=ci_image,
            noise_map=ci_noise_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            settings_dict=settings_dict,
        )

    def output_to_fits(
        self,
        data_path,
        noise_map_path=None,
        pre_cti_data_path=None,
        cosmic_ray_map_path=None,
        overwrite=False,
    ):
        self.image.output_to_fits(file_path=data_path, overwrite=overwrite)

        if noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)

        if pre_cti_data_path is not None:
            self.pre_cti_data.output_to_fits(
                file_path=pre_cti_data_path, overwrite=overwrite
            )

        if self.cosmic_ray_map is not None and cosmic_ray_map_path is not None:
            self.cosmic_ray_map.output_to_fits(
                file_path=cosmic_ray_map_path, overwrite=overwrite
            )
