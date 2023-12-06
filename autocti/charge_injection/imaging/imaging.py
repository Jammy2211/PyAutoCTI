import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union

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
        mask_persistence=None,
        noise_scaling_map_dict: Optional[Dict] = None,
        fpr_value: Optional[float] = None,
        settings_dict: Optional[Dict] = None,
    ):
        super().__init__(data=data, noise_map=noise_map)

        self.data = self.data.native
        self.noise_map = self.noise_map.native
        self.pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            cosmic_ray_map = cosmic_ray_map.native

        self.cosmic_ray_map = cosmic_ray_map
        self.mask_persistence = mask_persistence

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
        return self.data.mask

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
        masked_image = np.ma.array(data=self.data, mask=self.data.mask)

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
        image = aa.Array2D(values=self.data.native, mask=mask)
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
            mask_persistence=self.mask_persistence,
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
        pixel_scales: aa.type.PixelScales,
        layout: Layout2DCI,
        data_path: Optional[Union[Path, str]] = None,
        data_hdu: int = 0,
        data: aa.Array2D = None,
        noise_map_path: Optional[Union[Path, str]] = None,
        noise_map_hdu: int = 0,
        noise_map_from_single_value: float = None,
        pre_cti_data_path: Optional[Union[Path, str]] = None,
        pre_cti_data_hdu: int = 0,
        pre_cti_data: aa.Array2D = None,
        cosmic_ray_map_path: Optional[Union[Path, str]] = None,
        cosmic_ray_map_hdu: int = 0,
        settings_dict: Optional[Dict] = None,
    ) -> "ImagingCI":
        """
        Load charge injection imaging from multiple .fits file.

        For each attribute of the charge injection data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
        the .fits and the `hdu` containing the data can be specified.

        The `noise_map` assumes the noise value in each `data` value are independent, where these values are the
        RMS standard deviation error in each pixel.

        If the dataset has a mask associated with it (e.g. in a `mask.fits` file) the file must be loaded separately
        via the `Mask2D` object and applied to the imaging after loading via fits using the `from_fits` method.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        layout
            The layout of the charge injection, containing information like where the parallel and serial FPR and
            EPER are located.
        data_path
            The path to the data .fits file containing the image data (e.g. '/path/to/data.fits').
        data_hdu
            The hdu the image data is contained in the .fits file specified by `data_path`.
        data
            Manually input the data as an `Array2D` instead of loading it via a .fits file.
        noise_map_path
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits').
        noise_map_hdu
            The hdu the noise map is contained in the .fits file specified by `noise_map_path`.
        noise_map_from_single_value
            Creates a `noise_map` of constant values if this is input instead of loading via .fits.
        pre_cti_data_path
            The path to the pre CTI data .fits file containing the image data (e.g. '/path/to/pre_cti_data.fits').
        pre_cti_data_hdu
            The hdu the pre cti data is contained in the .fits file specified by `pre_cti_data_path`.
        pre_cti_data
            Manually input the pre CTI data as an `Array2D` instead of loading it via a .fits file.
        cosmic_ray_map_path
            The path to the cosmic ray map .fits file containing the map of cosmic
            rays (e.g. '/path/to/cosmic_ray_map.fits').
        cosmic_ray_map_hdu
            The hdu the cosmic ray data is contained in the .fits file specified by `cosmic_ray_map_path`.
        settings_dict
            A dictionary of settings associated with the charge injeciton imaging (e.g. voltage settings) which is
            used for visualization.
        """
        if data_path is not None and data is None:
            data = aa.Array2D.from_fits(
                file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
            )

        if noise_map_path is not None:
            noise_map = aa.util.array_2d.numpy_array_2d_via_fits_from(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            noise_map = np.ones(data.shape_native) * noise_map_from_single_value

        noise_map = aa.Array2D.no_mask(values=noise_map, pixel_scales=pixel_scales)

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
            data=data,
            noise_map=noise_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            settings_dict=settings_dict,
        )

    def output_to_fits(
        self,
        data_path: Union[Path, str],
        noise_map_path: Optional[Union[Path, str]] = None,
        pre_cti_data_path: Optional[Union[Path, str]] = None,
        cosmic_ray_map_path: Optional[Union[Path, str]] = None,
        overwrite: bool = False,
    ):
        """
        Output the charge injection imaging dataset to multiple .fits file.

        For each attribute of the charge injection imaging data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
        the .fits can be specified, with `hdu=0` assumed automatically.

        If the `data` has been masked, the masked data is output to .fits files. A mask can be separately output to
        a file `mask.fits` via the `Mask` objects `output_to_fits` method.

        Parameters
        ----------
        data_path
            The path to the data .fits file where the image data is output (e.g. '/path/to/data.fits').
        noise_map_path
            The path to the noise_map .fits where the noise_map is output (e.g. '/path/to/noise_map.fits').
        pre_cti_data_path
            The path to the pre CTI data .fits file where the pre CTI data is output (e.g. '/path/to/pre_cti_data.fits').
        cosmic_ray_map_path
            The path to the cosmic ray map .fits file where the cosmic ray map is
            output (e.g. '/path/to/cosmic_ray_map.fits').
        overwrite
            If `True`, the .fits files are overwritten if they already exist, if `False` they are not and an
            exception is raised.
        """
        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)

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
