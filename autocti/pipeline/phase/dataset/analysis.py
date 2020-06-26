from autocti import exc
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
        self,
        masked_ci_datasets,
        clocker,
        serial_total_density_range,
        parallel_total_density_range,
        results=None,
        pool=None,
    ):

        super().__init__()

        self.masked_ci_datasets = masked_ci_datasets
        self.clocker = clocker
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range
        self.pool = pool or ConsecutivePool

    def check_total_density_within_range(self, instance):

        if self.parallel_total_density_range is not None:

            total_density = sum([trap.density for trap in instance.parallel_traps])

            if (
                total_density < self.parallel_total_density_range[0]
                or total_density > self.parallel_total_density_range[1]
            ):
                raise exc.PriorException

        if self.serial_total_density_range is not None:

            total_density = sum([trap.density for trap in instance.serial_traps])

            if (
                total_density < self.serial_total_density_range[0]
                or total_density > self.serial_total_density_range[1]
            ):
                raise exc.PriorException
