from autofit.non_linear import abstract_search


class Result(abstract_search.Result):
    def __init__(self, samples, previous_model, analysis, search):
        """
        The result of a phase
        """
        super().__init__(samples=samples, previous_model=previous_model, search=search)

        self.analysis = analysis
        self.search = search

    @property
    def clocker(self):
        return self.analysis.clocker
