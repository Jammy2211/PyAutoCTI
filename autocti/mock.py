from autofit.tools.pipeline import ResultsCollection
from autofit import mock
from autofit.mock import MockSearch, MockSamples

import numpy as np

import autocti as ac

### Mock AutoFit ###


class MockResult(mock.MockResult):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
        mask=None,
        model_image=None,
        noise_scaling_maps_list_of_ci_regions=None,
        noise_scaling_maps_list_of_parallel_trails=None,
        noise_scaling_maps_list_of_serial_trails=None,
        noise_scaling_maps_list_of_serial_overscan_no_trails=None,
        use_as_hyper_dataset=False,
    ):

        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.previous_model = model
        self.gaussian_tuples = None
        self.mask = None
        self.positions = None
        self.mask = mask
        self.model_image = model_image
        self.noise_scaling_maps_list_of_ci_regions = (
            noise_scaling_maps_list_of_ci_regions
        )
        self.noise_scaling_maps_list_of_parallel_trails = (
            noise_scaling_maps_list_of_parallel_trails
        )
        self.noise_scaling_maps_list_of_serial_trails = (
            noise_scaling_maps_list_of_serial_trails
        )
        self.noise_scaling_maps_list_of_serial_overscan_no_trails = (
            noise_scaling_maps_list_of_serial_overscan_no_trails
        )
        self.hyper_combined = MockHyperCombinedPhase()
        self.use_as_hyper_dataset = use_as_hyper_dataset

    @property
    def last(self):
        return self


class MockResults(ResultsCollection):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        search=None,
        mask=None,
        model_image=None,
        noise_scaling_maps_list_of_ci_regions=None,
        noise_scaling_maps_list_of_parallel_trails=None,
        noise_scaling_maps_list_of_serial_trails=None,
        noise_scaling_maps_list_of_serial_overscan_no_trails=None,
        use_as_hyper_dataset=False,
    ):
        """
        A collection of results from previous phases. Results can be obtained using an index or the name of the phase
        from whence they came.
        """

        super().__init__()

        result = MockResult(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
            mask=mask,
            model_image=model_image,
            noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
            noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
            noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
            noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            use_as_hyper_dataset=use_as_hyper_dataset,
        )

        self.__result_list = [result]

    @property
    def last(self):
        """
        The result of the last phase
        """
        if len(self.__result_list) > 0:
            return self.__result_list[-1]
        return None

    def __getitem__(self, item):
        """
        Get the result of a previous phase by index

        Parameters
        ----------
        item: int
            The index of the result

        Returns
        -------
        result: Result
            The result of a previous phase
        """
        return self.__result_list[item]

    def __len__(self):
        return len(self.__result_list)


class MockHyperCombinedPhase:
    def __init__(self):
        pass


class MockPattern(object):
    def __init__(self, regions=None):

        self.normalization = 10
        self.regions = regions
        self.total_rows = 2
        self.total_columns = 2


class MockGeometry(object):
    def __init__(self):
        super(MockGeometry, self).__init__()


class MockFrameGeometry(object):
    def __init__(self, value=1.0):
        self.value = value

    def add_cti(self, image, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockCIFrame(object):
    def __init__(self, value=1.0):

        self.ci_pattern = MockPattern()
        self.frame_geometry = MockFrameGeometry(value=value)
        self.value = value

    def ci_regions_from_array(self, array):
        return array[0:2, 0]

    def parallel_non_ci_regions_frame_from_frame(self, array):
        return array[0:2, 1]

    def serial_all_trails_frame_from_frame(self, array):
        return array[0, 0:2]

    def serial_overscan_no_trails_frame_from_frame(self, array):
        return array[1, 0:2]

    def parallel_front_edge_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def parallel_trails_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def serial_front_edge_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def serial_trails_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])


class MockCIPreCTI(np.ndarray):
    def __new__(
        cls,
        array,
        frame_geometry=MockGeometry(),
        ci_pattern=MockPattern(),
        value=1.0,
        *args,
        **kwargs
    ):
        ci = np.array(array).view(cls)
        ci.frame_geometry = frame_geometry
        ci.ci_pattern = ci_pattern
        ci.value = value
        return ci

    def ci_post_cti_from_cti_params_and_settings(self, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockChInj(np.ndarray):
    def __new__(cls, array, geometry=None, ci_pattern=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = geometry
        ci.ci_pattern = ci_pattern
        return ci


### Arctic ###


def make_trap_0():
    return ac.TrapInstantCapture(density=10, release_timescale=-1 / np.log(0.5))


def make_trap_1():
    return ac.TrapInstantCapture(density=8, release_timescale=-1 / np.log(0.2))


def make_traps_x1():
    return [make_trap_0()]


def make_traps_x2():
    return [make_trap_0(), make_trap_1()]


def make_ccd():
    return ac.CCD(well_fill_power=0.5, full_well_depth=10000, well_notch_depth=1e-7)


def make_ccd_complex():
    return ac.CCDComplex(
        well_fill_alpha=1.0,
        well_fill_power=0.5,
        full_well_depth=10000,
        well_notch_depth=1e-7,
    )


def make_parallel_clocker():
    return ac.Clocker(
        parallel_express=2, parallel_charge_injection_mode=False, parallel_offset=0
    )


def make_serial_clocker():
    return ac.Clocker(serial_express=2, serial_offset=0)


### MASK ###


def make_mask_7x7():
    return ac.Mask2D.unmasked(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


### FRAMES ###


def make_scans_7x7():
    return ac.Scans(
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


def make_image_7x7():
    return ac.Frame.full(
        fill_value=1.0, shape_2d=(7, 7), scans=make_scans_7x7(), pixel_scales=(1.0, 1.0)
    )


def make_noise_map_7x7():
    return ac.Frame.full(
        fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0), scans=make_scans_7x7()
    )


### IMAGING ###


def make_imaging_7x7():
    return ac.Imaging(
        image=make_image_7x7(), noise_map=make_noise_map_7x7(), name="mock_imaging_7x7"
    )


### CHARGE INJECTION FRAMES ###


def make_ci_pattern_7x7():
    return ac.ci.CIPatternUniform(normalization=10.0, regions=[(1, 5, 1, 5)])


def make_ci_image_7x7():
    return ac.ci.CIFrame.full(
        fill_value=1.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        ci_pattern=make_ci_pattern_7x7(),
        roe_corner=(1, 0),
        scans=make_scans_7x7(),
    )


def make_ci_noise_map_7x7():
    return ac.ci.CIFrame.full(
        fill_value=2.0,
        shape_2d=(7, 7),
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_pre_cti_7x7():
    return ac.ci.CIFrame.full(
        shape_2d=(7, 7),
        fill_value=10.0,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_cosmic_ray_map_7x7():
    cosmic_ray_map = np.zeros(shape=(7, 7))

    return ac.ci.CIFrame.manual(
        array=cosmic_ray_map,
        pixel_scales=(1.0, 1.0),
        roe_corner=(1, 0),
        ci_pattern=make_ci_pattern_7x7(),
        scans=make_scans_7x7(),
    )


def make_ci_noise_scaling_maps_7x7():

    return [
        ac.ci.CIFrame.ones(
            shape_2d=(7, 7),
            pixel_scales=(1.0, 1.0),
            roe_corner=(1, 0),
            scans=make_scans_7x7(),
            ci_pattern=make_ci_pattern_7x7(),
        ),
        ac.ci.CIFrame.full(
            shape_2d=(7, 7),
            roe_corner=(1, 0),
            fill_value=2.0,
            scans=make_scans_7x7(),
            pixel_scales=(1.0, 1.0),
            ci_pattern=make_ci_pattern_7x7(),
        ),
    ]


### CHARGE INJECTION IMAGING ###


def make_ci_imaging_7x7():

    return ac.ci.CIImaging(
        image=make_ci_image_7x7(),
        noise_map=make_ci_noise_map_7x7(),
        ci_pre_cti=make_ci_pre_cti_7x7(),
        cosmic_ray_map=make_ci_cosmic_ray_map_7x7(),
    )


def make_masked_ci_imaging_7x7():
    return ac.ci.MaskedCIImaging(
        ci_imaging=make_ci_imaging_7x7(),
        mask=make_mask_7x7(),
        noise_scaling_maps=make_ci_noise_scaling_maps_7x7(),
    )


### CHARGE INJECTION FITS ###


def make_hyper_noise_scalars():
    return [
        ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
        ac.ci.CIHyperNoiseScalar(scale_factor=2.0),
    ]


def make_ci_fit_7x7():
    return ac.ci.CIFitImaging(
        masked_ci_imaging=make_masked_ci_imaging_7x7(),
        ci_post_cti=make_masked_ci_imaging_7x7().ci_pre_cti,
        hyper_noise_scalars=make_hyper_noise_scalars(),
    )


# ### PHASES ###

from autofit.mapper.model import ModelInstance


def make_samples_with_result():

    instance = ModelInstance()

    instance.parallel_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.parallel_ccd = make_ccd()
    instance.serial_traps = [ac.TrapInstantCapture(density=0, release_timescale=1)]
    instance.serial_ccd = make_ccd()

    instance.hyper_noise_scalar_of_ci_regions = None
    instance.hyper_noise_scalar_of_parallel_trails = None
    instance.hyper_noise_scalar_of_serial_trails = None
    instance.hyper_noise_scalar_of_serial_overscan_no_trails = None

    return mock.MockSamples(max_log_likelihood_instance=instance)


def make_phase_data():
    from autocti.pipeline.phase.dataset import PhaseDataset

    return PhaseDataset(search=mock.MockSearch(name="test_phase"))


def make_phase_ci_imaging_7x7():
    return ac.PhaseCIImaging(search=mock.MockSearch(name="test_phase"))


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2119))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
