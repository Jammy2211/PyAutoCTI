import autofit as af


def cti_params_for_instance(instance):
    return model.ArcticParams(
        parallel_ccd_volume=instance.parallel_ccd_volume
        if hasattr(instance, "parallel_ccd_volume")
        else None,
        parallel_traps=instance.parallel_traps
        if hasattr(instance, "parallel_traps")
        else None,
        serial_ccd_volume=instance.serial_ccd_volume
        if hasattr(instance, "serial_ccd_volume")
        else None,
        serial_traps=instance.serial_traps
        if hasattr(instance, "serial_traps")
        else None,
    )


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class Analysis(af.Analysis):
    def __init__(
        self,
        masked_ci_datasets,
        cti_settings,
        serial_total_density_range,
        parallel_total_density_range,
        results=None,
        pool=None,
    ):

        self.masked_ci_datasets = masked_ci_datasets
        self.cti_settings = cti_settings
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range
        self.pool = pool or ConsecutivePool

    def check_total_density_within_range(self, cti_params):

        if self.parallel_total_density_range is not None:

            total_density = sum(
                [
                    parallel_traps.trap_density
                    for parallel_traps in cti_params.parallel_traps
                ]
            )

            if (
                total_density < self.parallel_total_density_range[0]
                or total_density > self.parallel_total_density_range[1]
            ):
                raise exc.PriorException

        if self.serial_total_density_range is not None:

            total_density = sum(
                [serial_traps.trap_density for serial_traps in cti_params.serial_traps]
            )

            if (
                total_density < self.serial_total_density_range[0]
                or total_density > self.serial_total_density_range[1]
            ):
                raise exc.PriorException
