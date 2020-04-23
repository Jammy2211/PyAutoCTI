import autofit as af


class Result(af.Result):
    def __init__(self, samples, previous_model, analysis, optimizer):
        """
        The result of a phase
        """
        super().__init__(samples=samples, previous_model=previous_model)

        self.analysis = analysis
        self.optimizer = optimizer

    @property
    def clocker(self):
        return self.analysis.clocker
