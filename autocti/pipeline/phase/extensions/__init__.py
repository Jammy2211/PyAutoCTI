import autofit as af

from .hyper_noise_phase import HyperNoisePhase
from .hyper_phase import HyperPhase


class CombinedHyperPhase(HyperPhase):
    def __init__(self, phase, hyper_phase_classes: (type,) = tuple()):
        """
        A hyper_combined hyper_galaxy phase that can run zero or more other hyper_galaxy phases after the initial phase is
        run.

        Parameters
        ----------
        phase : phase.PhaseCIImaging
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
        datasets,
        clocker=None,
        pool=None,
        results: af.ResultsCollection = None,
        **kwargs
    ) -> af.Result:
        """
        Run the regular phase followed by the hyper_galaxy phases. Each result of a hyper_galaxy phase is attached to the
        overall result object by the hyper_name of that phase.

        Finally, a phase in run with all of the model results from all the individual hyper_galaxy phases.

        Parameters
        ----------
        positions
        mask
        datasets
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
        result = self.phase.run(datasets, results=results, clocker=clocker, **kwargs)
        results.add(self.phase.phase_name, result)

        for phase in self.hyper_phases:
            hyper_result = phase.run_hyper(
                datasets=datasets, pool=pool, results=results, **kwargs
            )
            setattr(result, phase.hyper_name, hyper_result)

        setattr(
            result,
            self.hyper_name,
            self.run_hyper(datasets=datasets, pool=pool, results=results),
        )
        return result

    def combine_variables(self, result) -> af.ModelMapper:
        """
        Combine the model objects from all previous results in this hyper_combined hyper_galaxy phase.

        Iterates through the hyper_galaxy names of the included hyper_galaxy phases, extracting a result
        for each name and adding the model of that result to a new model.

        Parameters
        ----------
        result
            The last result (with attribute results associated with phases in this phase)

        Returns
        -------
        combined_variable
            A model object including all variables from results in this phase.
        """
        model = af.ModelMapper()
        for name in self.phase_names:
            model += getattr(result, name).model
        return model

    def run_hyper(self, datasets, pool, results, **kwargs) -> af.Result:
        model = self.combine_variables(result=results.last)

        phase = self.make_hyper_phase()
        phase.optimizer.phase_tag = ""
        phase.optimizer.model = model

        phase.phase_tag = ""

        return phase.run(
            datasets=datasets, results=results, pool=pool, clocker=results.last.clocker
        )
