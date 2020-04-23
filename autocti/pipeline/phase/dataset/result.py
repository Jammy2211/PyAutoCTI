from autocti.pipeline.phase import abstract


class Result(abstract.result.Result):
    @property
    def max_log_likelihood_fits(self):
        return self.analysis.fits_from_instance(instance=self.instance)

    @property
    def masks(self):
        return [fit.mask for fit in self.max_log_likelihood_fits]
