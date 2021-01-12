from autocti.pipeline.phase.abstract.result import Result
from autofit.non_linear.paths import convert_paths
from autofit.tools import phase as af_phase


# noinspection PyAbstractClass


class AbstractPhase(af_phase.AbstractPhase):

    Result = Result

    def __init__(self, *, search):
        """
        A phase in an lens pipeline. Uses the set non_linear search to try to fit
        models and hyper_galaxies passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        name: str
            The name of this phase
        """

        super().__init__(search=search)

    def make_result(self, result, analysis):
        return self.Result(
            samples=result.samples,
            previous_model=result.model,
            analysis=analysis,
            search=self.search,
        )

    @property
    def is_parallel_fit(self):
        if self.model.parallel_ccd is not None and self.model.serial_ccd is None:
            return True
        else:
            return False

    @property
    def is_serial_fit(self):
        if self.model.parallel_ccd is None and self.model.serial_ccd is not None:
            return True
        else:
            return False

    @property
    def is_parallel_and_serial_fit(self):
        if self.model.parallel_ccd is not None and self.model.serial_ccd is not None:
            return True
        else:
            return False
