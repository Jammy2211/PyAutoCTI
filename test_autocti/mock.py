import autofit as af
import autocti as ac

import numpy as np


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
        ac.ci.frame_geometry = frame_geometry
        ac.ci.ci_pattern = ci_pattern
        ac.ci.value = value
        return ci

    def ci_post_cti_from_cti_params_and_settings(self, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockChInj(np.ndarray):
    def __new__(cls, array, geometry=None, ci_pattern=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ac.ci.frame_geometry = geometry
        ac.ci.ci_pattern = ci_pattern
        return ci


class MockSamples:
    def __init__(self, max_log_likelihood_instance=None):

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.log_likelihoods = [1.0]

    def gaussian_priors_at_sigma(self, sigma):
        return None

    @property
    def max_log_likelihood_instance(self):

        if self._max_log_likelihood_instance is not None:
            return self._max_log_likelihood_instance

        return af.ModelInstance()


class MockResult:
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        optimizer=None,
        mask=None,
        model_image=None,
        noise_scaling_maps_list_of_ci_regions=None,
        noise_scaling_maps_list_of_parallel_trails=None,
        noise_scaling_maps_list_of_serial_trails=None,
        noise_scaling_maps_list_of_serial_overscan_no_trails=None,
        use_as_hyper_dataset=False,
    ):

        self.instance = instance or af.ModelInstance()
        self.model = model or af.ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None
        self.positions = None
        self.mask_2d = mask
        self.model_image = model_image
        self.analysis = analysis
        self.optimizer = optimizer
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


class MockResults(af.ResultsCollection):
    def __init__(
        self,
        samples=None,
        instance=None,
        model=None,
        analysis=None,
        optimizer=None,
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
            optimizer=optimizer,
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


class MockNLO(af.NonLinearOptimizer):
    def _fit(self, analysis, fitness_function):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.log_likelihood_function(None), None)

    def _full_fit(self, model, analysis):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.log_likelihood_function(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def samples_from_model(self, model):
        return MockSamples()
