import numpy as np

from autoarray.exc import InversionException, GridException
from autofit.exc import FitException
from autocti.fit import fit
from autocti.pipeline import visualizer
from autocti.pipeline.phase.dataset import analysis as analysis_dataset


class Analysis(analysis_dataset.Analysis):
    def __init__(
        self,
        masked_ci_imagings,
        cti_settings,
        serial_total_density_range,
        parallel_total_density_range,
        image_path=None,
        results=None,
        pool=None,
    ):

        super().__init__(
            masked_ci_datasets=masked_ci_imagings,
            cti_settings=cti_settings,
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

        self.pool = pool

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
        cti_params = cti_params_for_instance(instance=instance)

        self.check_total_density_within_range(cti_params=cti_params)

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(instance=instance)

        pipe_cti_pass = partial(
            pipe_cti,
            cti_params=cti_params,
            cti_settings=self.cti_settings,
            hyper_noise_scalars=hyper_noise_scalars,
        )
        log_likelihood = np.sum(
            list(self.pool.map(pipe_cti_pass, self.masked_ci_datasets))
        )
        return log_likelihood

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

    def hyper_noise_scalars_from_instance_and_hyper_noise_scale(
        self, instance, hyper_noise_scale
    ):

        if hyper_noise_scale:
            return self.hyper_noise_scalars_from_instance(instance=instance)
        else:
            return None

    def fits_of_ci_data_extracted_for_instance(self, instance, hyper_noise_scale=True):

        cti_params = cti_params_for_instance(instance=instance)
        hyper_noise_scalars = self.hyper_noise_scalars_from_instance_and_hyper_noise_scale(
            instance=instance, hyper_noise_scale=hyper_noise_scale
        )

        return list(
            map(
                lambda ci_data_masked_extracted: ci_fit.CIFitImaging(
                    masked_ci_imaging=ci_data_masked_extracted,
                    cti_params=cti_params,
                    cti_settings=self.cti_settings,
                    hyper_noise_scalars=hyper_noise_scalars,
                ),
                self.masked_ci_datasets,
            )
        )

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
                    cti_settings=self.cti_settings,
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


def pipe_cti(ci_data_masked, cti_params, cti_settings, hyper_noise_scalars):
    fit = ci_fit.CIFitImaging(
        masked_ci_imaging=ci_data_masked,
        cti_params=cti_params,
        cti_settings=cti_settings,
        hyper_noise_scalars=hyper_noise_scalars,
    )
    return fit.figure_of_merit
