import copy

from autoconf import conf
from autofit.optimize import non_linear
from autofit.tools.pipeline import ResultsCollection


class HyperPhase(object):
    def __init__(self, phase, hyper_name: str):
        """
        Abstract HyperPhase. Wraps a phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A phase
        """
        self.phase = phase
        self.hyper_name = hyper_name

    def run_hyper(self, *args, **kwargs) -> non_linear.Result:
        """
        Run the hyper_galaxies phase.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        result
            The result of the hyper_galaxies phase.
        """
        raise NotImplementedError()

    def make_hyper_phase(self):
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """

        phase = copy.deepcopy(self.phase)

        phase_folders = phase.phase_folders
        phase_folders.append(phase.phase_name)

        phase.optimizer = phase.optimizer.copy_with_name_extension(
            extension=self.hyper_name + "_" + phase.phase_tag
        )

        phase.optimizer.const_efficiency_mode = conf.instance.non_linear.get(
            "MultiNest", "const_efficiency_mode", bool
        )
        phase.optimizer.sampling_efficiency = conf.instance.non_linear.get(
            "MultiNest", "sampling_efficiency", float
        )
        phase.optimizer.n_live_points = conf.instance.non_linear.get(
            "MultiNest", "n_live_points", int
        )
        phase.optimizer.multimodal = conf.instance.non_linear.get(
            "MultiNest", "multimodal", bool
        )

        phase.is_hyper_phase = True
        phase.optimizer.phase_tag = ""
        phase.phase_tag = ""
        phase.customize_priors = self.customize_priors

        return phase

    def customize_priors(self, results):

        pass

    def run(
        self, datasets, results: ResultsCollection = None, **kwargs
    ) -> non_linear.Result:
        """
        Run the hyper phase and then the hyper_galaxies phase.

        Parameters
        ----------
        datasets
            Data
        results
            Results from previous phases.
        kwargs

        Returns
        -------
        result
            The result of the phase, with a hyper_galaxies result attached as an attribute with the hyper_name of this
            phase.
        """

        results = copy.deepcopy(results) if results is not None else ResultsCollection()

        result = self.phase.run(
            datasets=datasets, clocker=results.last.clocker, results=results, **kwargs
        )
        results.add(self.phase.phase_name, result)
        hyper_result = self.run_hyper(ci_data=datasets, results=results, **kwargs)
        setattr(result, self.hyper_name, hyper_result)
        return result

    def __getattr__(self, item):
        return getattr(self.phase, item)
