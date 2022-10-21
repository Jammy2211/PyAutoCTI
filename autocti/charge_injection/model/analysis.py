from typing import Optional, List

from autofit.non_linear.samples import PDFSamples
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.paths.directory import DirectoryPaths

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.fit import FitImagingCI
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.clocker.two_d import Clocker2D
from autocti.model.settings import SettingsCTI2D
from autocti.preloads import Preloads

from autocti import exc


class AnalysisImagingCI(Analysis):
    def __init__(
        self, dataset: ImagingCI, clocker: Clocker2D, settings_cti=SettingsCTI2D()
    ):

        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti

        self.preloads = Preloads()

        parallel_fast_index_list = None
        parallel_fast_column_lists = None

        serial_fast_index_list = None
        serial_fast_row_lists = None

        if self.clocker.parallel_fast_mode and not self.clocker.serial_fast_mode:

            (
                parallel_fast_index_list,
                parallel_fast_column_lists,
            ) = clocker.fast_indexes_from(data=dataset.pre_cti_data, for_parallel=True)

        elif not self.clocker.parallel_fast_mode and self.clocker.serial_fast_mode:

            serial_fast_index_list, serial_fast_row_lists = clocker.fast_indexes_from(
                data=dataset.pre_cti_data, for_parallel=False
            )

        elif self.clocker.parallel_fast_mode and self.clocker.serial_fast_mode:

            raise exc.ClockerException(
                "Both parallel fast model and serial fast mode cannot be turned on.\n"
                "Only switch on parallel fast mode for parallel + serial clocking."
            )

        self.preloads = Preloads(
            parallel_fast_index_list=parallel_fast_index_list,
            parallel_fast_column_lists=parallel_fast_column_lists,
            serial_fast_index_list=serial_fast_index_list,
            serial_fast_row_lists=serial_fast_row_lists,
        )

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
            parallel_traps=instance.cti.parallel_trap_list,
            serial_traps=instance.cti.serial_trap_list,
        )

        fit = self.fit_via_instance_and_dataset_from(
            instance=instance, imaging_ci=self.dataset, hyper_noise_scale=True
        )

        return fit.figure_of_merit

    def fit_via_instance_and_dataset_from(
        self,
        instance: ModelInstance,
        imaging_ci: ImagingCI,
        hyper_noise_scale: bool = True,
    ) -> FitImagingCI:

        hyper_noise_scalar_dict = None

        if hyper_noise_scale and hasattr(instance, "hyper_noise"):
            hyper_noise_scalar_dict = instance.hyper_noise.as_dict

        post_cti_data = self.clocker.add_cti(
            data=imaging_ci.pre_cti_data, cti=instance.cti, preloads=self.preloads
        )

        return FitImagingCI(
            dataset=imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalar_dict=hyper_noise_scalar_dict,
        )

    def fit_via_instance_from(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_via_instance_and_dataset_from(
            instance=instance,
            imaging_ci=self.dataset,
            hyper_noise_scale=hyper_noise_scale,
        )

    def visualize(
        self, paths: DirectoryPaths, instance: ModelInstance, during_analysis: bool
    ):

        fit = self.fit_via_instance_from(instance=instance)

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

        visualizer.visualize_imaging_ci(imaging_ci=self.dataset)

        try:
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset, region="parallel_fpr"
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset, region="parallel_eper"
            )

            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="parallel_fpr"
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="parallel_eper"
            )
        except (exc.RegionException, TypeError, ValueError):
            pass

        try:
            visualizer.visualize_imaging_ci(imaging_ci=self.dataset)
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset, region="serial_fpr"
            )
            visualizer.visualize_imaging_ci_lines(
                imaging_ci=self.dataset, region="serial_eper"
            )

            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="serial_fpr"
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="serial_eper"
            )
        except (exc.RegionException, TypeError, ValueError):
            pass

    def make_result(
        self,
        samples: PDFSamples,
        model: CollectionPriorModel,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultImagingCI:
        return ResultImagingCI(samples=samples, model=model, analysis=self)
