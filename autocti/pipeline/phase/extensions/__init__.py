import autofit as af
from autocti.pipeline.phase import phase_ci as ph
from .hyper_noise_phase import HyperNoisePhase
from .hyper_phase import HyperPhase


class CombinedHyperPhase(HyperPhase):
    def __init__(self, phase: ph.PhaseCI, hyper_phase_classes: (type,) = tuple()):
        """
        A hyper_combined hyper_galaxy phase that can run zero or more other hyper_galaxy phases after the initial phase is
        run.

        Parameters
        ----------
        phase : phase.PhaseCI
            The phase wrapped by this hyper_galaxy phase
        hyper_phase_classes
            The classes of hyper_galaxy phases to be run following the initial phase
        """
        super().__init__(phase, "hyper_combined")
        self.hyper_phases = list(map(lambda cls: cls(phase), hyper_phase_classes))

    @property
    def phase_names(self) -> [str]:
        """
        The names of phases included in this hyper_combined phase
        """
        return [phase.hyper_name for phase in self.hyper_phases]

    def run(
        self,
        ci_datas,
        cti_settings=None,
        pool=None,
        results: af.ResultsCollection = None,
        **kwargs
    ) -> af.Result:
        """
        Run the regular phase followed by the hyper_galaxy phases. Each result of a hyper_galaxy phase is attached to the
        overall result object by the hyper_name of that phase.

        Finally, a phase in run with all of the variable results from all the individual hyper_galaxy phases.

        Parameters
        ----------
        positions
        mask
        ci_datas
            The instrument
        results
            Results from previous phases
        kwargs

        Returns
        -------
        result
            The result of the regular phase, with hyper_galaxy results attached by associated hyper_galaxy names
        """
        results = results.copy() if results is not None else af.ResultsCollection()
        result = self.phase.run(
            ci_datas, results=results, cti_settings=cti_settings, **kwargs
        )
        results.add(self.phase.phase_name, result)

        for phase in self.hyper_phases:
            hyper_result = phase.run_hyper(
                ci_datas=ci_datas, pool=pool, results=results, **kwargs
            )
            setattr(result, phase.hyper_name, hyper_result)

        setattr(
            result,
            self.hyper_name,
            self.run_hyper(ci_datas=ci_datas, pool=pool, results=results),
        )
        return result

    def combine_variables(self, result) -> af.ModelMapper:
        """
        Combine the variable objects from all previous results in this hyper_combined hyper_galaxy phase.

        Iterates through the hyper_galaxy names of the included hyper_galaxy phases, extracting a result
        for each name and adding the variable of that result to a new variable.

        Parameters
        ----------
        result
            The last result (with attribute results associated with phases in this phase)

        Returns
        -------
        combined_variable
            A variable object including all variables from results in this phase.
        """
        variable = af.ModelMapper()
        for name in self.phase_names:
            variable += getattr(result, name).variable
        return variable

    def run_hyper(self, ci_datas, pool, results, **kwargs) -> af.Result:
        variable = self.combine_variables(result=results.last)

        phase = self.make_hyper_phase()
        phase.optimizer.phase_tag = ""
        phase.optimizer.variable = variable

        phase.phase_tag = ""

        return phase.run(
            ci_datas=ci_datas,
            results=results,
            pool=pool,
            cti_settings=results.last.cti_settings,
        )
