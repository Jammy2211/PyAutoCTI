class HyperCINoise(object):

    def __init__(self, scale_factor=0.0):
        """The ci_hyper-parameter factor by which the noises is scaled when included in the model-fitting process"""
        self.scale_factor = scale_factor

    def scaled_noise_map_from_noise_scaling(self, noise_scaling):
        """Compute the scaled noises map, by multiplying the noises-scaling image by the ci_hyper-parameter factor."""
        return self.scale_factor * noise_scaling

    def __repr__(self):
        return "Noise Scale Factor: {}".format(self.scale_factor) + '\n'
