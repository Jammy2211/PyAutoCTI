from typing import Optional, List

from autofit.non_linear.samples import PDFSamples
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.abstract_search import NonLinearSearch

from autocti.charge_injection.imaging import ImagingCI
from autocti.charge_injection.fit import FitImagingCI
from autocti.charge_injection.hyper import HyperCINoiseScalar
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autocti.model.result import ResultDataset
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.model.settings import SettingsCTI2D
from autocti.util.clocker import Clocker2D

from autocti import exc


class AnalysisImagingCI(Analysis):
    def __init__(
        self,
        dataset_ci: ImagingCI,
        clocker: Clocker2D,
        settings_cti=SettingsCTI2D(),
        results: List[ResultDataset] = None,
    ):

        super().__init__()

        self.dataset_ci = dataset_ci
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results

    def log_likelihood_function(self, instance: ModelInstance) -> float:
        """
        Determine the fitness of a particular model

        Parameters
        ----------
        instance

        Returns
        -------
        fit: Fit
            How fit the model is and the model
        """

        self.settings_cti.check_total_density_within_range(
            parallel_traps=instance.cti.parallel_traps,
            serial_traps=instance.cti.serial_traps,
        )

        hyper_noise_scalars = self.hyper_noise_scalars_from_instance(instance=instance)

        if instance.cti.parallel_traps is not None:
            parallel_traps = list(instance.cti.parallel_traps)
        else:
            parallel_traps = None

        if instance.cti.serial_traps is not None:
            serial_traps = list(instance.cti.serial_traps)
        else:
            serial_traps = None

        post_cti_data = self.clocker.add_cti(
            data=self.dataset_ci.pre_cti_data,
            parallel_trap_list=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_trap_list=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        fit = FitImagingCI(
            dataset=self.dataset_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=hyper_noise_scalars,
        )

        return fit.log_likelihood

    def hyper_noise_scalars_from_instance(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> Optional[List[HyperCINoiseScalar]]:

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
        self,
        instance: ModelInstance,
        imaging_ci: ImagingCI,
        hyper_noise_scale: bool = True,
    ) -> FitImagingCI:

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

        post_cti_data = self.clocker.add_cti(
            data=imaging_ci.pre_cti_data,
            parallel_trap_list=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_trap_list=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        return FitImagingCI(
            dataset=imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=hyper_noise_scalars,
        )

    def fit_from_instance(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_from_instance_and_imaging_ci(
            instance=instance,
            imaging_ci=self.dataset_ci,
            hyper_noise_scale=hyper_noise_scale,
        )

    def fit_full_dataset_from_instance(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_from_instance_and_imaging_ci(
            instance=instance,
            imaging_ci=self.dataset_ci.imaging_full,
            hyper_noise_scale=hyper_noise_scale,
        )

    def visualize(
        self, paths: DirectoryPaths, instance: ModelInstance, during_analysis: bool
    ):

        fit = self.fit_from_instance(instance=instance)

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

        visualizer.visualize_imaging_ci(imaging_ci=self.dataset_ci)

        try:
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset_ci, line_region="parallel_front_edge"
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset_ci, line_region="parallel_trails"
            )

            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit,
                during_analysis=during_analysis,
                line_region="parallel_front_edge",
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, line_region="parallel_trails"
            )
        except exc.RegionException:
            pass

        try:
            visualizer.visualize_imaging_ci(imaging_ci=self.dataset_ci)
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset_ci, line_region="serial_front_edge"
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset_ci, line_region="serial_trails"
            )

            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit,
                during_analysis=during_analysis,
                line_region="serial_front_edge",
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, line_region="serial_trails"
            )
        except exc.RegionException:
            pass

    def make_result(
        self, samples: PDFSamples, model: CollectionPriorModel, search: NonLinearSearch
    ) -> ResultImagingCI:
        return ResultImagingCI(
            samples=samples, model=model, analysis=self, search=search
        )
