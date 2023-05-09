from typing import Optional


class HyperCINoiseScalar(float):
    def __new__(cls, scale_factor=0.0):
        return super().__new__(cls, scale_factor)

    def __init__(self, scale_factor=0.0):
        """
        The hyper_ci-parameter factor by which the noises is scaled when included in the model-fitting process.
        """
        float.__init__(scale_factor)

    def scaled_noise_map_from(self, noise_scaling):
        """
        Returns the scaled noises map, by multiplying the noises-scaling image by the hyper_ci-parameter factor.
        """
        return self * noise_scaling


class HyperCINoiseCollection:
    def __init__(
        self,
        regions_ci: Optional[HyperCINoiseScalar] = None,
        parallel_eper: Optional[HyperCINoiseScalar] = None,
        serial_eper: Optional[HyperCINoiseScalar] = None,
        serial_overscan_no_eper: Optional[HyperCINoiseScalar] = None,
    ):
        self.regions_ci = regions_ci
        self.parallel_eper = parallel_eper
        self.serial_eper = serial_eper
        self.serial_overscan_no_eper = serial_overscan_no_eper

    @property
    def as_dict(self):
        hyper_dict = {
            "regions_ci": self.regions_ci,
            "parallel_eper": self.parallel_eper,
            "serial_eper": self.serial_eper,
            "serial_overscan_no_eper": self.serial_overscan_no_eper,
        }

        return {key: value for key, value in hyper_dict.items() if value is not None}
