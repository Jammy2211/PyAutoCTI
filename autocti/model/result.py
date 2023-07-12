from autofit.non_linear import result


class Result(result.Result):
    def __init__(self, samples, analysis):
        """
        The result of a phase
        """
        super().__init__(samples=samples)

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
