from autocti.pipeline.phase.abstract import Result as AbstractResult


class Result(AbstractResult):
    @property
    def max_log_likelihood_fits(self):
        return self.analysis.fits_from_instance(instance=self.instance)

    @property
    def masks(self):
        return [fit.mask for fit in self.max_log_likelihood_fits]
