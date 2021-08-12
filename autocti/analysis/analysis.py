from typing import Optional, List

import autofit as af
from autofit.non_linear.abstract_search import Analysis
from autocti.charge_injection.imaging_ci import ImagingCI
from autocti.charge_injection.fit_ci import FitImagingCI
from autocti.charge_injection.hyper_ci import HyperCINoiseScalar
from autocti.line.dataset_line import DatasetLine
from autocti.line.fit_line import FitDatasetLine
from autocti.analysis import visualizer as vis
from autocti.analysis.result import ResultDataset
from autocti.analysis.result import ResultDatasetLine
from autocti.analysis.result import ResultImagingCI
from autocti.analysis.settings import SettingsCTI1D
from autocti.analysis.settings import SettingsCTI2D
from autocti.util.clocker import Clocker1D
from autocti.util.clocker import Clocker2D


class AnalysisDatasetLine(Analysis):
    def __init__(
        self,
        dataset_line: DatasetLine,
        clocker: Clocker1D,
        settings_cti: SettingsCTI1D = SettingsCTI1D(),
        results: List[ResultDataset] = None,
    ):

        super().__init__()

        self.dataset_line = dataset_line
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
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

        self.settings_cti.check_total_density_within_range(traps=instance.cti.traps)

        fit = self.fit_from_instance(instance=instance)

        return fit.log_likelihood

    def fit_from_instance_and_dataset_line(
        self, instance: af.ModelInstance, dataset_line: DatasetLine
    ) -> FitDatasetLine:

        if instance.cti.traps is not None:
            traps = list(instance.cti.traps)
        else:
            traps = None

        post_cti_line = self.clocker.add_cti(
            pre_cti_data=dataset_line.pre_cti_line, traps=traps, ccd=instance.cti.ccd
        )

        return FitDatasetLine(dataset_line=dataset_line, post_cti_line=post_cti_line)

    def fit_from_instance(self, instance: af.ModelInstance) -> FitDatasetLine:

        return self.fit_from_instance_and_dataset_line(
            instance=instance, dataset_line=self.dataset_line
        )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        fit = self.fit_from_instance(instance=instance)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ) -> ResultDatasetLine:
        return ResultDatasetLine(
            samples=samples, model=model, analysis=self, search=search
        )


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

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
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

        post_cti_image = self.clocker.add_cti(
            pre_cti_data=self.dataset_ci.pre_cti_image,
            parallel_traps=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        fit = FitImagingCI(
            imaging=self.dataset_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=hyper_noise_scalars,
        )

        return fit.log_likelihood

    def hyper_noise_scalars_from_instance(
        self, instance: af.ModelInstance, hyper_noise_scale: bool = True
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
        instance: af.ModelInstance,
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

        post_cti_image = self.clocker.add_cti(
            pre_cti_data=imaging_ci.pre_cti_image,
            parallel_traps=parallel_traps,
            parallel_ccd=instance.cti.parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=instance.cti.serial_ccd,
        )

        return FitImagingCI(
            imaging=imaging_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=hyper_noise_scalars,
        )

    def fit_from_instance(
        self, instance: af.ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_from_instance_and_imaging_ci(
            instance=instance,
            imaging_ci=self.dataset_ci,
            hyper_noise_scale=hyper_noise_scale,
        )

    def fit_full_dataset_from_instance(
        self, instance: af.ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_from_instance_and_imaging_ci(
            instance=instance,
            imaging_ci=self.dataset_ci.imaging_full,
            hyper_noise_scale=hyper_noise_scale,
        )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        fit = self.fit_from_instance(instance=instance)

        visualizer = vis.Visualizer(visualize_path=paths.image_path)

        visualizer.visualize_imaging_ci(imaging_ci=self.dataset_ci)
        visualizer.visualize_imaging_ci_lines(
            imaging_ci=self.dataset_ci, line_region="parallel_front_edge"
        )
        visualizer.visualize_imaging_ci_lines(
            imaging_ci=self.dataset_ci, line_region="parallel_trails"
        )

        visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_fit_ci_1d_lines(
            fit=fit, during_analysis=during_analysis, line_region="parallel_front_edge"
        )
        visualizer.visualize_fit_ci_1d_lines(
            fit=fit, during_analysis=during_analysis, line_region="parallel_trails"
        )

        # visualizer.visualize_multiple_fit_cis_subplots(fits=fit)
        # visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
        #     fits=fit, line_region="parallel_front_edge"
        # )
        # visualizer.visualize_multiple_fit_cis_subplots_1d_lines(
        #     fits=fit, line_region="parallel_trails"
        # )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ) -> ResultImagingCI:
        return ResultImagingCI(
            samples=samples, model=model, analysis=self, search=search
        )
