from functools import partial

import numpy as np
from autocti.charge_injection import ci_fit
from autocti.pipeline import visualizer
from autocti.pipeline.phase.dataset import analysis as analysis_dataset


class Analysis(analysis_dataset.Analysis):
    def __init__(
        self,
        masked_ci_imagings,
        clocker,
        settings_cti,
        image_path=None,
        results=None,
        pool=None,
    ):

        super().__init__(
            masked_ci_datasets=masked_ci_imagings,
            clocker=clocker,
            settings_cti=settings_cti,
            results=results,
        )

        self.visualizers = [
            visualizer.PhaseCIImagingVisualizer(
                masked_dataset=masked_ci_imaging, image_path=image_path, results=results
            )
            for masked_ci_imaging in masked_ci_imagings
        ]

        self.pool = pool or analysis_dataset.ConsecutivePool

    @property
    def masked_ci_imagings(self):
        return self.masked_ci_datasets

    def log_likelihood_function(self, instance):
        """
        Determine the fitness of a particular model

        Parameters
        ----------
        instance

        Returns
        -------
        fit: ci_fit.Fit
            How fit the model is and the model
        """

        self.settings_cti.check_total_density_within_range(
            parallel_traps=instance.parallel_traps, serial_traps=instance.serial_traps
        )

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(instance=instance)

        pipe_cti_pass = partial(
            pipe_cti,
            instance=instance,
            clocker=self.clocker,
            hyper_noise_scalars=hyper_noise_scalars,
        )

        return np.sum(list(self.pool.map(pipe_cti_pass, self.masked_ci_datasets)))

    def hyper_noise_scalars_from_instance(self, instance, hyper_noise_scale=True):

        if not hyper_noise_scale:
            return None

        hyper_noise_scalars = list(
            filter(
                None,
                [
                    instance.hyper_noise_scalar_of_ci_regions,
                    instance.hyper_noise_scalar_of_parallel_trails,
                    instance.hyper_noise_scalar_of_serial_trails,
                    instance.hyper_noise_scalar_of_serial_overscan_no_trails,
                ],
            )
        )

        if hyper_noise_scalars:
            return hyper_noise_scalars

    def fit_from_instance_and_ci_imaging(
        self, instance, ci_imaging, hyper_noise_scale=True
    ):

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(
            instance=instance, hyper_noise_scale=hyper_noise_scale
        )

        if len(instance.parallel_traps) > 0:
            parallel_traps = list(instance.parallel_traps)
        else:
            parallel_traps = None

        if len(instance.serial_traps) > 0:
            serial_traps = list(instance.serial_traps)
        else:
            serial_traps = None

        ci_post_cti = self.clocker.add_cti(
            image=ci_imaging.ci_pre_cti,
            parallel_traps=parallel_traps,
            parallel_ccd=instance.parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=instance.serial_ccd,
        )

        return ci_fit.CIFitImaging(
            masked_ci_imaging=ci_imaging,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=hyper_noise_scalars,
        )

    def fits_from_instance(self, instance, hyper_noise_scale=True):

        return [
            self.fit_from_instance_and_ci_imaging(
                instance=instance,
                ci_imaging=masked_ci_imaging,
                hyper_noise_scale=hyper_noise_scale,
            )
            for masked_ci_imaging in self.masked_ci_imagings
        ]

    def fits_full_dataset_from_instance(self, instance, hyper_noise_scale=True):

        return [
            self.fit_from_instance_and_ci_imaging(
                instance=instance,
                ci_imaging=masked_ci_imaging.ci_imaging_full,
                hyper_noise_scale=hyper_noise_scale,
            )
            for masked_ci_imaging in self.masked_ci_imagings
        ]

    def visualize(self, instance, during_analysis):

        fits = self.fits_from_instance(instance=instance)

        for fit, visualizer in zip(fits, self.visualizers):

            visualizer.visualize_ci_fit(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_ci_fit_lines(fit=fit, during_analysis=during_analysis)


def pipe_cti(ci_data_masked, instance, clocker, hyper_noise_scalars):

    # TODO : Convesions ini pyarctic make this dodgy - will fix but sorting them out in arcticpy.

    if len(instance.parallel_traps) > 0:
        parallel_traps = list(instance.parallel_traps)
    else:
        parallel_traps = None

    if len(instance.serial_traps) > 0:
        serial_traps = list(instance.serial_traps)
    else:
        serial_traps = None

    ci_post_cti = clocker.add_cti(
        image=ci_data_masked.ci_pre_cti,
        parallel_traps=parallel_traps,
        parallel_ccd=instance.parallel_ccd,
        serial_traps=serial_traps,
        serial_ccd=instance.serial_ccd,
    )

    fit = ci_fit.CIFitImaging(
        masked_ci_imaging=ci_data_masked,
        ci_post_cti=ci_post_cti,
        hyper_noise_scalars=hyper_noise_scalars,
    )

    return fit.log_likelihood
