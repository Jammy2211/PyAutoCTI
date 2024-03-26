from autofit.non_linear import result


class Result(result.Result):
    def __init__(self, samples_summary, paths=None, samples=None, analysis=None, search_internal = None):
        """
        The result of a phase
        """
        super().__init__(samples_summary=samples_summary, paths=paths, samples=samples, search_internal=search_internal)

        self.analysis = analysis

    @property
    def clocker(self):
        return self.analysis.clocker


class ResultDataset(Result):
    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_via_instance_from(instance=self.instance)

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask
