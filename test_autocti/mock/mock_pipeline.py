import numpy as np

import autofit as af
import autocti as al


class GalaxiesMockAnalysis:
    def __init__(self, number_galaxies, value):
        self.number_galaxies = number_galaxies
        self.value = value
        self.hyper_model_image = None
        self.hyper_galaxy_image_path_dict = None

    # noinspection PyUnusedLocal
    def galaxy_images_for_model(self, model):
        return self.number_galaxies * [np.array([self.value])]

    def fit(self, instance):
        return 1


class MockResult:
    def __init__(
        self,
        instance=None,
        log_likelihood=None,
        model=None,
        analysis=None,
        optimizer=None,
        mask=None,
        model_image=None,
        use_as_hyper_dataset=False,
    ):
        self.instance = instance
        self.log_likelihood = log_likelihood
        self.model = model
        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None
        self.positions = None
        self.mask_2d = mask
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.instance = instance or af.ModelInstance()
        self.model = af.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer
        self.hyper_combined = MockHyperCombinedPhase()
        self.use_as_hyper_dataset = use_as_hyper_dataset

    @property
    def last(self):
        return self


class MockResults(af.ResultsCollection):
    def __init__(
        self,
        instance=None,
        log_likelihood=None,
        analysis=None,
        optimizer=None,
        model=None,
        mask=None,
        model_image=None,
        use_as_hyper_dataset=False,
    ):
        """
        A collection of results from previous phases. Results can be obtained using an index or the name of the phase
        from whence they came.
        """

        super().__init__()

        result = MockResult(
            instance=instance,
            log_likelihood=log_likelihood,
            analysis=analysis,
            optimizer=optimizer,
            model=model,
            mask=mask,
            model_image=model_image,
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
                self.result = MockResult(instance, log_likelihood)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def samples_from_model(self, model, paths):
        return MockOutput()


class MockOutput:
    def __init__(self):
        pass
