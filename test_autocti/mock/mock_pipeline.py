import numpy as np

import autofit as af
import autocti as al


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
    def _simple_fit(self, analysis, fitness_function):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.fit(None), None)

    def _fit(self, analysis, model):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.fit(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def samples_from_model(self, model):
        return MockSamples()
