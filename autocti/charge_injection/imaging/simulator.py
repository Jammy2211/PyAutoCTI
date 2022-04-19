import numpy as np

import autoarray as aa

from autoarray.dataset.imaging import AbstractSimulatorImaging

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.layout import Layout2DCI
from autocti.clocker.two_d import Clocker2D
from autocti.model.model_util import CTI2D

from typing import Optional


class SimulatorImagingCI(AbstractSimulatorImaging):
    def __init__(
        self,
        pixel_scales: aa.type.PixelScales,
        norm: float,
        max_norm: float = np.inf,
        column_sigma: Optional[float] = None,
        row_slope: Optional[float] = 0.0,
        read_noise: Optional[float] = None,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
        ci_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map
            The exposure time of an observation using this data_type.
        """

        super().__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.pixel_scales = pixel_scales

        self.norm = norm
        self.max_norm = max_norm
        self.column_sigma = column_sigma
        self.row_slope = row_slope

        self.ci_seed = ci_seed

    @property
    def _ci_seed(self):

        if self.ci_seed == -1:
            return np.random.randint(0, int(1e9))
        return self.ci_seed

    def region_ci_from(self, region_dimensions):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each pre_cti_datas \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulator this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform charge injection images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        max_normalization
        column_sigma
        region_dimensions
            The size of the non-uniform charge injection region.
        ci_seed
            Input seed for the random number generator to give reproducible results.
        """

        np.random.seed(self._ci_seed)

        ci_rows = region_dimensions[0]
        ci_columns = region_dimensions[1]
        ci_region = np.zeros(region_dimensions)

        for column_number in range(ci_columns):

            column_norm = 0
            while column_norm <= 0 or column_norm >= self.max_norm:
                column_norm = np.random.normal(self.norm, self.column_sigma)

            ci_region[0:ci_rows, column_number] = self.generate_column(
                size=ci_rows, norm=column_norm, row_slope=self.row_slope
            )

        return ci_region

    def pre_cti_data_uniform_from(self, layout: Layout2DCI):
        """
        Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        """

        pre_cti_data = np.zeros(layout.shape_2d)

        for region in layout.region_list:
            pre_cti_data[region.slice] += self.norm

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=self.pixel_scales)

    def pre_cti_data_non_uniform_from(self, layout: Layout2DCI) -> aa.Array2D:
        """
        Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection pre_cti_datas, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        A non-uniform charge injection layout_ci, which is defined by the regions it appears on a charge injection
        array and its average normalization.

        Non-uniformity across the columns of a charge injection layout_ci is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope row_slope.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        ci_seed
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_data = np.zeros(layout.shape_2d)

        for region in layout.region_list:
            pre_cti_data[region.slice] += self.region_ci_from(
                region_dimensions=region.shape
            )

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=self.pixel_scales)

    def generate_column(self, size: int, norm: float, row_slope: float) -> np.ndarray:
        """
        Generate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

        Parameters
        -----------
        size
            The size of the non-uniform column of charge
        normalization
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return norm * (np.arange(1, size + 1)) ** row_slope

    def via_layout_from(
        self,
        layout: Layout2DCI,
        clocker: Clocker2D,
        cti: CTI2D,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        pre_cti_data
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout
            The charge injection layout_ci (regions, normalization, etc.) of the charge injection image.
        cti_params
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed
            Seed for the read-noises added to the image.
        """

        if self.column_sigma is not None:
            pre_cti_data = self.pre_cti_data_non_uniform_from(layout=layout)
        else:
            pre_cti_data = self.pre_cti_data_uniform_from(layout=layout)

        return self.via_pre_cti_data_from(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=clocker,
            cti=cti,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def via_pre_cti_data_from(
        self,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        clocker: Clocker2D,
        cti: CTI2D,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

        pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            pre_cti_data += cosmic_ray_map.native

        post_cti_data = clocker.add_cti(data=pre_cti_data, cti=cti)

        if cosmic_ray_map is not None:
            pre_cti_data -= cosmic_ray_map.native

        return self.via_post_cti_data_from(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def via_post_cti_data_from(
        self,
        post_cti_data: aa.Array2D,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

        if self.read_noise is not None:
            ci_image = aa.preprocess.data_with_gaussian_noise_added(
                data=post_cti_data, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = (
                self.read_noise
                * aa.Array2D.ones(
                    shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            ci_image = post_cti_data
            ci_noise_map = aa.Array2D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_2d,
                pixel_scales=self.pixel_scales,
            ).native

        return ImagingCI(
            image=aa.Array2D.manual(
                array=ci_image.native, pixel_scales=self.pixel_scales
            ),
            noise_map=aa.Array2D.manual(
                array=ci_noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_data=aa.Array2D.manual(
                array=pre_cti_data.native, pixel_scales=self.pixel_scales
            ),
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            name=name,
        )
