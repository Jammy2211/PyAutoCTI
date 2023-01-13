import numpy as np
from typing import Optional

import autoarray as aa

from autoarray.dataset.imaging.simulator import SimulatorImaging
from autoarray.dataset import preprocess

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.layout.one_d import Layout1D
from autocti.clocker.one_d import Clocker1D
from autocti.model.model_util import CTI1D


class SimulatorDataset1D(SimulatorImaging):
    def __init__(
        self,
        pixel_scales: aa.type.PixelScales,
        norm: float,
        read_noise: Optional[float] = None,
        add_poisson_noise: bool = False,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """
        A class representing a Imaging observation, using the shape of the data, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map
            The exposure time of an observation using this data_type.
        """

        super().__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.pixel_scales = pixel_scales
        self.norm = norm

    def pre_cti_data_from(
        self, layout: Layout1D, pixel_scales: aa.type.PixelScales
    ) -> aa.Array1D:
        """Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        ----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        """

        pre_cti_data = np.zeros(layout.shape_1d)

        for region in layout.region_list:
            pre_cti_data[region.slice] += self.norm

        return aa.Array1D.no_mask(values=pre_cti_data, pixel_scales=pixel_scales)

    def via_layout_from(
        self, layout: Layout1D, clocker: Clocker1D, cti: CTI1D
    ) -> Dataset1D:
        """Simulate a charge injection data, including effects like noises.

        Parameters
        ----------
        pre_cti_data
        cosmic_ray_map
            The dimensions of the output simulated charge injection data.
        frame_geometry
            The quadrant geometry of the simulated data, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout : layout_ci.Layout1DSimulate
            The charge injection layout_ci (regions, normalization, etc.) of the charge injection data.
        cti_params
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise
            The FWHM of the Gaussian read-noises added to the data.
        noise_seed
            Seed for the read-noises added to the data.
        """

        pre_cti_data = self.pre_cti_data_from(
            layout=layout, pixel_scales=self.pixel_scales
        )

        return self.via_pre_cti_data_from(
            pre_cti_data=pre_cti_data.native, layout=layout, clocker=clocker, cti=cti
        )

    def via_pre_cti_data_from(
        self, pre_cti_data: aa.Array1D, layout: Layout1D, clocker: Clocker1D, cti: CTI1D
    ) -> Dataset1D:

        post_cti_data = clocker.add_cti(data=pre_cti_data.native, cti=cti)

        return self.via_post_cti_data_from(
            post_cti_data=post_cti_data, pre_cti_data=pre_cti_data, layout=layout
        )

    def via_post_cti_data_from(
        self, post_cti_data: aa.Array1D, pre_cti_data: aa.Array1D, layout: Layout1D
    ) -> Dataset1D:

        if self.read_noise is not None:
            data = preprocess.data_with_gaussian_noise_added(
                data=post_cti_data, sigma=self.read_noise, seed=self.noise_seed
            )
            noise_map = (
                self.read_noise
                * aa.Array1D.ones(
                    shape_native=layout.shape_1d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            data = post_cti_data
            noise_map = aa.Array1D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_1d,
                pixel_scales=self.pixel_scales,
            ).native

        return Dataset1D(
            data=aa.Array1D.no_mask(
                values=data.native, pixel_scales=self.pixel_scales
            ),
            noise_map=aa.Array1D.no_mask(
                values=noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_data=aa.Array1D.no_mask(
                values=pre_cti_data.native, pixel_scales=self.pixel_scales
            ),
            layout=layout,
        )
