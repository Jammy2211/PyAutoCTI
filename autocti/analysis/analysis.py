from functools import partial
import numpy as np

from autofit.non_linear import abstract_search
from autocti.charge_injection import ci_fit, ci_frame
from autocti.analysis import visualizer as vis, settings


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
        ci_datasets,
        clocker,
        settings_cti=settings.SettingsCTI(),
        results=None,
        pool=None,
    ):

        super().__init__()

        self.ci_datasets = ci_datasets
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results
        self.pool = pool or ConsecutivePool


class AnalysisCIImaging(Analysis):
    def __init__(
        self,
        dataset_list,
        clocker,
        settings_cti=settings.SettingsCTI(),
        results=None,
        pool=None,
    ):

        super().__init__(
            ci_datasets=dataset_list,
            clocker=clocker,
            settings_cti=settings_cti,
            results=results,
        )

        self.pool = pool or ConsecutivePool

    @property
    def ci_imagings(self):
        return self.ci_datasets

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
            parallel_traps=instance.cti.parallel_traps,
            serial_traps=instance.cti.serial_traps,
        )

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(instance=instance)

        pipe_cti_pass = partial(
            pipe_cti,
            instance=instance,
            clocker=self.clocker,
            hyper_noise_scalars=hyper_noise_scalars,
        )

        return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datasets)))

    def hyper_noise_scalars_from_instance(self, instance, hyper_noise_scale=True):

        if not hasattr(instance, "hyper_noise"):
            return None

        if not hyper_noise_scale:
            return None

        hyper_noise_scalars = list(
            filter(
                None,
                [
                    instance.hyper_noise.ci_regions,
                    instance.hyper_noise.parallel_trails,
                    instance.hyper_noise.serial_trails,
                    instance.hyper_noise.serial_overscan_no_trails,
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

        if instance.cti.parallel_traps is not None:
            parallel_traps = list(instance.cti.parallel_traps)
        else:
            parallel_traps = None

        if instance.cti.serial_traps is not None:
            serial_traps = list(instance.cti.serial_traps)
        else:
            serial_traps = None

        ci_post_cti = self.clocker.add_cti(
            image=ci_imaging.ci_pre_cti,
            parallel_traps=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        ci_post_cti = ci_frame.CIFrame.manual(
            array=ci_post_cti,
            pixel_scales=ci_imaging.ci_pre_cti.pixel_scales,
            ci_pattern=ci_imaging.ci_pre_cti.ci_pattern,
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
                ci_imaging=ci_imaging,
                hyper_noise_scale=hyper_noise_scale,
            )
            for ci_imaging in self.ci_imagings
        ]

    def fits_full_dataset_from_instance(self, instance, hyper_noise_scale=True):

        return [
            self.fit_from_instance_and_ci_imaging(
                instance=instance,
                ci_imaging=ci_imaging.ci_imaging_full,
                hyper_noise_scale=hyper_noise_scale,
            )
            for ci_imaging in self.ci_imagings
        ]

    def visualize(self, paths, instance, during_analysis):

        fits = self.fits_from_instance(instance=instance)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        for index in range(len(fits)):

            visualizer.visualize_ci_imaging(
                ci_imaging=self.ci_datasets[index], index=index
            )
            visualizer.visualize_ci_imaging_lines(
                ci_imaging=self.ci_datasets[index],
                line_region="parallel_front_edge",
                index=index,
            )
            visualizer.visualize_ci_imaging_lines(
                ci_imaging=self.ci_datasets[index],
                line_region="parallel_trails",
                index=index,
            )

            visualizer.visualize_ci_fit(
                fit=fits[index], during_analysis=during_analysis, index=index
            )
            visualizer.visualize_ci_fit_1d_lines(
                fit=fits[index],
                during_analysis=during_analysis,
                line_region="parallel_front_edge",
                index=index,
            )
            visualizer.visualize_ci_fit_1d_lines(
                fit=fits[index],
                during_analysis=during_analysis,
                line_region="parallel_trails",
                index=index,
            )

        visualizer.visualize_multiple_ci_fits_subplots(fits=fits)
        visualizer.visualize_multiple_ci_fits_subplots_1d_lines(
            fits=fits, line_region="parallel_front_edge"
        )
        visualizer.visualize_multiple_ci_fits_subplots_1d_lines(
            fits=fits, line_region="parallel_trails"
        )


def pipe_cti(ci_data_masked, instance, clocker, hyper_noise_scalars):

    # TODO : Convesions ini pyarctic make this dodgy - will fix but sorting them out in arcticpy.

    if instance.cti.parallel_traps is not None:
        parallel_traps = list(instance.cti.parallel_traps)
    else:
        parallel_traps = None

    if instance.cti.serial_traps is not None:
        serial_traps = list(instance.cti.serial_traps)
    else:
        serial_traps = None

    ci_post_cti = clocker.add_cti(
        image=ci_data_masked.ci_pre_cti,
        parallel_traps=parallel_traps,
        parallel_ccd=instance.cti.parallel_ccd,
        serial_traps=serial_traps,
        serial_ccd=instance.cti.serial_ccd,
    )

    fit = ci_fit.CIFitImaging(
        masked_ci_imaging=ci_data_masked,
        ci_post_cti=ci_post_cti,
        hyper_noise_scalars=hyper_noise_scalars,
    )

    return fit.log_likelihood
