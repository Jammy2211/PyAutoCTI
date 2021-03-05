from autofit.non_linear import abstract_search


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class Analysis(abstract_search.Analysis):
    def __init__(
        self, masked_ci_datasets, clocker, settings_cti, results=None, pool=None
    ):

        super().__init__()

        self.masked_ci_datasets = masked_ci_datasets
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.pool = pool or ConsecutivePool
