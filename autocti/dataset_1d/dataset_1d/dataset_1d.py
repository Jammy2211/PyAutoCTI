import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union

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
        pixel_scales: aa.type.PixelScales,
        layout: Layout1D,
        data_path: Optional[Union[Path, str]] = None,
        data_hdu: int = 0,
        noise_map_path: Optional[Union[Path, str]] = None,
        noise_map_hdu: int = 0,
        noise_map_from_single_value: float = None,
        pre_cti_data_path: Optional[Union[Path, str]] = None,
        pre_cti_data_hdu: int = 0,
        pre_cti_data: aa.Array1D = None,
        settings_dict: Optional[Dict] = None,
    ):
        """
        Load 1D dataset from multiple .fits file.

        For each attribute of the 1D dataset (e.g. `data`, `noise_map`, `pre_cti_data`) the path to the .fits and
        the `hdu` containing the data can be specified.

        The `noise_map` assumes the noise value in each `data` value are independent, where these values are the
        RMS standard deviation error in each pixel.

        If the dataset has a mask associated with it (e.g. in a `mask.fits` file) the file must be loaded separately
        via the `Mask1D` object and applied to the imaging after loading via fits using the `from_fits` method.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        layout
            The layout of the 1D dataset, containing information like where the FPR and EPER are located.
        data_path
            The path to the data .fits file containing the data (e.g. '/path/to/data.fits').
        data_hdu
            The hdu the image data is contained in the .fits file specified by `data_path`.
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
            Manually input the pre CTI data as an `Array1D` instead of loading it via a .fits file.
        settings_dict
            A dictionary of settings associated with the charge injeciton imaging (e.g. voltage settings) which is
            used for visualization.
        """
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
        self,
        data_path: Union[Path, str],
        noise_map_path: Optional[Union[Path, str]] = None,
        pre_cti_data_path: Optional[Union[Path, str]] = None,
        overwrite: bool = False,
    ):
        """
        Output the 1D dataset to multiple .fits file.

        For each attribute of the 1D dataset data (e.g. `data`, `noise_map`, `pre_cti_data`) the path to
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
        overwrite
            If `True`, the .fits files are overwritten if they already exist, if `False` they are not and an
            exception is raised.
        """
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
