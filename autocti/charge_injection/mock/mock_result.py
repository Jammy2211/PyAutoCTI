from autofit import mock
from autofit.non_linear.result import ResultsCollection


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
        noise_scaling_map_list_list_of_parallel_epers=None,
        noise_scaling_map_list_list_of_serial_epers=None,
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
        self.noise_scaling_map_list_list_of_parallel_epers = (
            noise_scaling_map_list_list_of_parallel_epers
        )
        self.noise_scaling_map_list_list_of_serial_epers = (
            noise_scaling_map_list_list_of_serial_epers
        )
        self.noise_scaling_map_list_list_of_serial_overscan_no_trails = (
            noise_scaling_map_list_list_of_serial_overscan_no_trails
        )
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
        noise_scaling_map_list_list_of_parallel_epers=None,
        noise_scaling_map_list_list_of_serial_epers=None,
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
            noise_scaling_map_list_list_of_parallel_epers=noise_scaling_map_list_list_of_parallel_epers,
            noise_scaling_map_list_list_of_serial_epers=noise_scaling_map_list_list_of_serial_epers,
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
