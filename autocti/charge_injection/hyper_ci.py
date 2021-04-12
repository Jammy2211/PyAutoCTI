from typing import Optional


class CIHyperNoiseScalar(float):
    def __new__(cls, scale_factor=0.0):
        return super().__new__(cls, scale_factor)

    def __init__(self, scale_factor=0.0):
        """The hyper_ci-parameter factor by which the noises is scaled when included in the model-fitting process"""
        float.__init__(scale_factor)

    def scaled_noise_map_from_noise_scaling(self, noise_scaling):
        """
        Returns the scaled noises map, by multiplying the noises-scaling image by the hyper_ci-parameter factor."""
        return self * noise_scaling

    def __repr__(self):
        return "Noise Scale Factor: {}".format(self) + "\n"


class CIHyperNoiseCollection:
    def __init__(
        self,
        ci_regions: Optional[CIHyperNoiseScalar] = None,
        parallel_trails: Optional[CIHyperNoiseScalar] = None,
        serial_trails: Optional[CIHyperNoiseScalar] = None,
        serial_overscan_no_trails: Optional[CIHyperNoiseScalar] = None,
    ):

        self.ci_regions = ci_regions
        self.parallel_trails = parallel_trails
        self.serial_trails = serial_trails
        self.serial_overscan_no_trails = serial_overscan_no_trails
