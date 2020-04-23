import numpy as np
from functools import partial

from autoarray.exc import InversionException, GridException
from autofit.exc import FitException
from autocti.charge_injection import ci_fit
from autocti.pipeline import visualizer
from autocti.pipeline.phase.dataset import analysis as analysis_dataset


class Analysis(analysis_dataset.Analysis):
    def __init__(
        self,
        masked_ci_imagings,
        clocker,
        serial_total_density_range,
        parallel_total_density_range,
        image_path=None,
        results=None,
        pool=None,
    ):

        super().__init__(
            masked_ci_datasets=masked_ci_imagings,
            clocker=clocker,
            parallel_total_density_range=parallel_total_density_range,
            serial_total_density_range=serial_total_density_range,
            results=results,
        )

        self.visualizer = [
            visualizer.PhaseCIImagingVisualizer(
                masked_dataset=masked_ci_imaging, image_path=image_path, results=results
            )
            for masked_ci_imaging in masked_ci_imagings
        ]

        self.pool = pool or analysis_dataset.ConsecutivePool

    @property
    def masked_ci_imagings(self):
        return self.masked_ci_datasets

    def fit(self, instance):
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

        self.check_total_density_within_range(instance=instance)

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(instance=instance)

        pipe_cti_pass = partial(
            pipe_cti,
            instance=instance,
            clocker=self.clocker,
            hyper_noise_scalars=hyper_noise_scalars,
        )

        return np.sum(list(self.pool.map(pipe_cti_pass, self.masked_ci_datasets)))

    def hyper_noise_scalars_from_instance(self, instance):

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
        else:
            return None

    def fits_from_instance(self, instance, hyper_noise_scale=True):

        if hyper_noise_scale:
            hyper_noise_scalars = self.hyper_noise_scalars_from_instance(
                instance=instance
            )
        else:
            hyper_noise_scalars = None

        ci_post_ctis = [
            self.clocker.add_cti(
                image=masked_ci_imaging.ci_pre_cti,
                parallel_traps=instance.parallel_traps,
                parallel_ccd_volume=instance.parallel_ccd_volume,
                serial_traps=instance.serial_traps,
                serial_ccd_volume=instance.serial_ccd_volume,
            )
            for masked_ci_imaging in self.masked_ci_imagings
        ]

        return [
            ci_fit.CIFitImaging(
                masked_ci_imaging=masked_ci_imaging,
                ci_post_cti=ci_post_cti,
                hyper_noise_scalars=hyper_noise_scalars,
            )
            for masked_ci_imaging, ci_post_cti in zip(
                self.masked_ci_imagings, ci_post_ctis
            )
        ]

    def fits_of_ci_data_full_for_instance(self, instance, hyper_noise_scale=True):

        cti_params = cti_params_for_instance(instance=instance)
        hyper_noise_scalars = self.hyper_noise_scalars_from_instance_and_hyper_noise_scale(
            instance=instance, hyper_noise_scale=hyper_noise_scale
        )

        return list(
            map(
                lambda ci_data_masked_full: ci_fit.CIFitImaging(
                    masked_ci_imaging=ci_data_masked_full,
                    cti_params=cti_params,
                    clocker=self.clocker,
                    hyper_noise_scalars=hyper_noise_scalars,
                ),
                self.masked_ci_dataset_full,
            )
        )

    def visualize(self, instance, during_analysis):

        if not self.is_hyper:
            fits = self.fits_of_ci_data_extracted_for_instance(instance=instance)
        elif self.is_hyper:
            fits = self.fits_of_ci_data_hyper_extracted_for_instance(instance=instance)

        self.visualizer.visualize_ci_fit(fit=fit, during_analysis=during_analysis)
        self.visualizer.visualize_ci_fit_lines(fit=fit, during_analysis=during_analysis)
        self.visualizer.visualize_multiple_ci_fits_subplots(
            fits=fits, during_analysis=during_analysis
        )
        self.visualizer.visualize_multiple_ci_fits_subplots_lines(
            fits=fits, during_analysis=during_analysis
        )


def pipe_cti(ci_data_masked, instance, clocker, hyper_noise_scalars):

    print(instance.parallel_traps)
    print(instance.parallel_ccd_volume)
    print(instance.serial_traps)

    ci_post_cti = clocker.add_cti(
        image=ci_data_masked.ci_pre_cti,
        parallel_traps=instance.parallel_traps,
        parallel_ccd_volume=instance.parallel_ccd_volume,
        serial_traps=instance.serial_traps,
        serial_ccd_volume=instance.serial_ccd_volume,
    )

    fit = ci_fit.CIFitImaging(
        masked_ci_imaging=ci_data_masked,
        ci_post_cti=ci_post_cti,
        hyper_noise_scalars=hyper_noise_scalars,
    )

    return fit.log_likelihood
