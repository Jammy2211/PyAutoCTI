import autofit as af


class AbstractPhase(af.AbstractPhase):
    def __init__(
        self,
        phase_name,
        phase_tag=None,
        phase_folders=tuple(),
        non_linear_class=af.MultiNest,
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        phase_name: str
            The name of this phase
        """

        self.phase_folders = phase_folders

        super().__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            non_linear_class=non_linear_class,
        )

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def customize_priors(self, results):
        """
        Perform any prior or constant passing. This could involve setting model
        attributes equal to priors or constants from a previous phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous phase
        """
        pass

    # noinspection PyAbstractClass
    class Analysis(af.Analysis):
        def __init__(self, results=None):
            """
            An lens object

            Parameters
            ----------
            results: autofit.tools.pipeline.ResultsCollection
                The results of all previous phases
            """

            self.results = results

            self.plot_count = 0

        @property
        def last_results(self):
            """
            Returns
            -------
            result: AbstractPhase.Result | None
                The result from the last phase
            """
            if self.results is not None:
                return self.results.last

    def make_result(self, result, analysis):
        return self.__class__.Result(
            constant=result.constant,
            log_likelihood=result.figure_of_merit,
            previous_variable=result.previous_variable,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            optimizer=self.optimizer,
        )

    class Result(af.Result):
        def __init__(
            self,
            constant,
            log_likelihood,
            previous_variable,
            gaussian_tuples,
            analysis,
            optimizer,
        ):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(
                constant=constant,
                log_likelihood=log_likelihood,
                previous_variable=previous_variable,
                gaussian_tuples=gaussian_tuples,
            )

            self.analysis = analysis
            self.optimizer = optimizer

        @property
        def cti_settings(self):
            return self.analysis.cti_settings


class Phase(AbstractPhase):
    def run(self, ci_datas, cti_settings, results=None, pool=None):
        raise NotImplementedError()

    # noinspection PyAbstractClass
    class Analysis(AbstractPhase.Analysis):
        def __init__(self, results=None):
            super(Phase.Analysis, self).__init__(results=results)
