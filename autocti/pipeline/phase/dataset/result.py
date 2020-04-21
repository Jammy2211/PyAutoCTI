from autocti.pipeline.phase import abstract


class Result(abstract.result.Result):
    @property
    def mask(self):
        return self.most_likely_fit.mask
