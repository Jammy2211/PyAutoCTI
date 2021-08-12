import numpy as np

from autofit.mock import mock
from autofit.non_linear.result import ResultsCollection
from autofit.mock.mock import MockSamples
from autofit.mock.mock import MockSearch

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
        noise_scaling_map_list_list_of_regions_ci=None,
        noise_scaling_map_list_list_of_parallel_trails=None,
        noise_scaling_map_list_list_of_serial_trails=None,
        noise_scaling_map_list_list_of_serial_overscan_no_trails=None,
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
        self.noise_scaling_map_list_list_of_regions_ci = (
            noise_scaling_map_list_list_of_regions_ci
        )
        self.noise_scaling_map_list_list_of_parallel_trails = (
            noise_scaling_map_list_list_of_parallel_trails
        )
        self.noise_scaling_map_list_list_of_serial_trails = (
            noise_scaling_map_list_list_of_serial_trails
        )
        self.noise_scaling_map_list_list_of_serial_overscan_no_trails = (
            noise_scaling_map_list_list_of_serial_overscan_no_trails
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
        noise_scaling_map_list_list_of_regions_ci=None,
        noise_scaling_map_list_list_of_parallel_trails=None,
        noise_scaling_map_list_list_of_serial_trails=None,
        noise_scaling_map_list_list_of_serial_overscan_no_trails=None,
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
            noise_scaling_map_list_list_of_regions_ci=noise_scaling_map_list_list_of_regions_ci,
            noise_scaling_map_list_list_of_parallel_trails=noise_scaling_map_list_list_of_parallel_trails,
            noise_scaling_map_list_list_of_serial_trails=noise_scaling_map_list_list_of_serial_trails,
            noise_scaling_map_list_list_of_serial_overscan_no_trails=noise_scaling_map_list_list_of_serial_overscan_no_trails,
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
    def __init__(self, region_list=None):

        self.normalization = 10
        self.region_list = region_list
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

        self.layout_ci = MockPattern()
        self.frame_geometry = MockFrameGeometry(value=value)
        self.value = value

    def regions_ci_from_array(self, array):
        return array[0:2, 0]

    def parallel_non_frame_with_extracted_regions_ci_from_from_frame(self, array):
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
        layout_ci=MockPattern(),
        value=1.0,
        *args,
        **kwargs
    ):
        ci = np.array(array).view(cls)
        ci.frame_geometry = frame_geometry
        ci.layout_ci = layout_ci
        ci.value = value
        return ci

    def post_cti_data_from_cti_params_and_settings(self, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockChInj(np.ndarray):
    def __new__(cls, array, geometry=None, layout_ci=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = geometry
        ci.layout_ci = layout_ci
        return ci
