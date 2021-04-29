from functools import partial
import numpy as np

from autofit.non_linear import abstract_search
from autoarray.structures.arrays.two_d import array_2d
from autocti.charge_injection import fit_ci
from autocti.analysis import visualizer as vis, settings


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class AnalysisImagingCI(abstract_search.Analysis):
    def __init__(
        self,
        ci_dataset_list,
        clocker,
        settings_cti=settings.SettingsCTI(),
        results=None,
        pool=None,
    ):

        super().__init__()

        self.ci_dataset_list = ci_dataset_list
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results
        self.pool = pool or ConsecutivePool

    @property
    def imaging_ci_list(self):
        return self.ci_dataset_list

    def log_likelihood_function(self, instance):
        """
        Determine the fitness of a particular model

        Parameters
        ----------
        instance

        Returns
        -------
        fit: fit_ci.Fit
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

        return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_dataset_list)))

    def hyper_noise_scalars_from_instance(self, instance, hyper_noise_scale=True):

        if not hasattr(instance, "hyper_noise"):
            return None

        if not hyper_noise_scale:
            return None

        hyper_noise_scalars = list(
            filter(
                None,
                [
                    instance.hyper_noise.regions_ci,
                    instance.hyper_noise.parallel_trails,
                    instance.hyper_noise.serial_trails,
                    instance.hyper_noise.serial_overscan_no_trails,
                ],
            )
        )

        if hyper_noise_scalars:
            return hyper_noise_scalars

    def fit_from_instance_and_imaging_ci(
        self, instance, imaging_ci, hyper_noise_scale=True
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

        post_cti_ci = self.clocker.add_cti(
            image=imaging_ci.pre_cti_ci,
            parallel_traps=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        return fit_ci.FitImagingCI(
            imaging_ci=imaging_ci,
            post_cti_ci=post_cti_ci,
            hyper_noise_scalars=hyper_noise_scalars,
        )

    def fit_list_from_instance(self, instance, hyper_noise_scale=True):

        return [
            self.fit_from_instance_and_imaging_ci(
                instance=instance,
                imaging_ci=imaging_ci,
                hyper_noise_scale=hyper_noise_scale,
            )
            for imaging_ci in self.imaging_ci_list
        ]

    def fits_full_dataset_from_instance(self, instance, hyper_noise_scale=True):

        return [
            self.fit_from_instance_and_imaging_ci(
                instance=instance,
                imaging_ci=imaging_ci.imaging_ci_full,
                hyper_noise_scale=hyper_noise_scale,
            )
            for imaging_ci in self.imaging_ci_list
        ]

    def visualize(self, paths, instance, during_analysis):

        fits = self.fit_list_from_instance(instance=instance)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        for index in range(len(fits)):

            visualizer.visualize_imaging_ci(
                imaging_ci=self.ci_dataset_list[index], index=index
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.ci_dataset_list[index],
                line_region="parallel_front_edge",
                index=index,
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.ci_dataset_list[index],
                line_region="parallel_trails",
                index=index,
            )

            visualizer.visualize_fit_ci(
                fit=fits[index], during_analysis=during_analysis, index=index
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fits[index],
                during_analysis=during_analysis,
                line_region="parallel_front_edge",
                index=index,
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fits[index],
                during_analysis=during_analysis,
                line_region="parallel_trails",
                index=index,
            )

        visualizer.visualize_multiple_fit_cis_subplots(fits=fits)
        visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
            fits=fits, line_region="parallel_front_edge"
        )
        visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
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

    post_cti_ci = clocker.add_cti(
        image=ci_data_masked.pre_cti_ci,
        parallel_traps=parallel_traps,
        parallel_ccd=instance.cti.parallel_ccd,
        serial_traps=serial_traps,
        serial_ccd=instance.cti.serial_ccd,
    )

    fit = fit_ci.FitImagingCI(
        imaging_ci=ci_data_masked,
        post_cti_ci=post_cti_ci,
        hyper_noise_scalars=hyper_noise_scalars,
    )

    return fit.log_likelihood
