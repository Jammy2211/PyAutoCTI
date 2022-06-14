from typing import Optional, List

from arcticpy.src import ccd
from arcticpy.src import traps

from autofit.non_linear.samples import PDFSamples
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.abstract_search import NonLinearSearch

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.fit import FitImagingCI
from autocti.charge_injection.hyper import HyperCINoiseScalar
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.clocker.two_d import Clocker2D
from autocti.model.result import ResultDataset
from autocti.model.model_util import CTI2D
from autocti.model.settings import SettingsCTI2D
from autocti.preloads import Preloads

from autocti import exc


class AnalysisImagingCI(Analysis):
    def __init__(
        self,
        dataset: ImagingCI,
        clocker: Clocker2D,
        settings_cti=SettingsCTI2D(),
        results: List[ResultDataset] = None,
    ):

        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results

        self.preloads = Preloads()

        parallel_fast_index_list = None
        parallel_fast_column_lists = None

        serial_fast_index_list = None
        serial_fast_row_lists = None

        if self.clocker.parallel_fast_mode and not self.clocker.serial_fast_mode:

            parallel_fast_index_list, parallel_fast_column_lists = clocker.fast_indexes_from(
                data=dataset.pre_cti_data, for_parallel=True
            )

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

        hyper_noise_scalar_list = self.hyper_noise_scalar_list_from(instance=instance)

        post_cti_data = self.clocker.add_cti(
            data=self.dataset.pre_cti_data, cti=instance.cti, preloads=self.preloads
        )

        fit = FitImagingCI(
            dataset=self.dataset,
            post_cti_data=post_cti_data,
            hyper_noise_scalar_list=hyper_noise_scalar_list,
        )

        return fit.log_likelihood

    def hyper_noise_scalar_list_from(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> Optional[List[HyperCINoiseScalar]]:

        if not hasattr(instance, "hyper_noise"):
            return None

        if not hyper_noise_scale:
            return None

        return [
            instance.hyper_noise.regions_ci,
            instance.hyper_noise.parallel_epers,
            instance.hyper_noise.serial_eper,
            instance.hyper_noise.serial_overscan_no_trails,
        ]

        # hyper_noise_scalar_list = list(
        #     filter(
        #         None,
        #         [
        #             instance.hyper_noise.regions_ci,
        #             instance.hyper_noise.parallel_epers,
        #             instance.hyper_noise.serial_eper,
        #             instance.hyper_noise.serial_overscan_no_trails,
        #         ],
        #     )
        # )

        if hyper_noise_scalar_list:
            return hyper_noise_scalar_list

    def fit_via_instance_and_dataset_from(
        self,
        instance: ModelInstance,
        imaging_ci: ImagingCI,
        hyper_noise_scale: bool = True,
    ) -> FitImagingCI:

        hyper_noise_scalar_list = self.hyper_noise_scalar_list_from(
            instance=instance, hyper_noise_scale=hyper_noise_scale
        )

        post_cti_data = self.clocker.add_cti(
            data=imaging_ci.pre_cti_data, cti=instance.cti, preloads=self.preloads
        )

        return FitImagingCI(
            dataset=imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalar_list=hyper_noise_scalar_list,
        )

    def fit_via_instance_from(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_via_instance_and_dataset_from(
            instance=instance,
            imaging_ci=self.dataset,
            hyper_noise_scale=hyper_noise_scale,
        )

    def fit_full_dataset_via_instance_from(
        self, instance: ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_via_instance_and_dataset_from(
            instance=instance,
            imaging_ci=self.dataset.imaging_full,
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
