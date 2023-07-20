import logging
import os
from typing import List, Optional

from autoconf import conf

import autoarray as aa
import autofit as af

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.fit import FitImagingCI
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.clocker.two_d import Clocker2D
from autocti.charge_injection.hyper import HyperCINoiseCollection
from autocti.model.settings import SettingsCTI2D
from autocti.preloads import Preloads

from autocti import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisImagingCI(af.Analysis):
    def __init__(
        self,
        dataset: ImagingCI,
        clocker: Clocker2D,
        settings_cti: SettingsCTI2D = SettingsCTI2D(),
        dataset_full: Optional[ImagingCI] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.dataset_full = dataset_full

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

    def region_list_from(self, model: af.Collection) -> List:
        """
        Inspects the CTI model and determines which regions are fitted for and therefore should be visualized.

        For example, if the model only includes parallel CTI, the serial regions are not fitted for and thus are not
        visualized.

        Parameters
        ----------
        model
            The CTI model, composed via PyAutoFit, which represents the parallel and serial CTI model compoenents
            fitted for by the non-linear search.

        Returns
        -------
        A list of the regions fitted for by the model and therefore visualized.

        """
        if model.cti.serial_ccd is None:
            return ["parallel_fpr", "parallel_eper"]
        elif model.cti.parallel_ccd is None:
            return ["serial_fpr", "serial_eper"]
        return ["parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"]

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

        This function:

         1) Visualizes the charge injection imaging dataset, which does not change during the analysis and thus can be
            done once.

         2) Checks if the noise-map is fixed (it is not if hyper functionality is on), and if it is fixed it
            sets the noise-normalization to the preloads for computational speed.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        if paths.is_complete:
            return self

        if not model.has(HyperCINoiseCollection):
            noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
                noise_map=self.dataset.noise_map, mask=self.dataset.mask
            )

            self.preloads.noise_normalization = noise_normalization

            logger.info(
                "PRELOADS - Noise Normalization preloaded for model-fit (noise-map is fixed)."
            )

        return self

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
            parallel_traps=instance.cti.parallel_trap_list,
            serial_traps=instance.cti.serial_trap_list,
        )

        fit = self.fit_via_instance_and_dataset_from(
            instance=instance, dataset=self.dataset, hyper_noise_scale=True
        )

        return fit.figure_of_merit

    def fit_via_instance_and_dataset_from(
        self,
        instance: af.ModelInstance,
        dataset: ImagingCI,
        hyper_noise_scale: bool = True,
    ) -> FitImagingCI:
        hyper_noise_scalar_dict = None

        if hyper_noise_scale and hasattr(instance, "hyper_noise"):
            hyper_noise_scalar_dict = instance.hyper_noise.as_dict

        post_cti_data = self.clocker.add_cti(
            data=dataset.pre_cti_data, cti=instance.cti, preloads=self.preloads
        )

        return FitImagingCI(
            dataset=dataset,
            post_cti_data=post_cti_data,
            hyper_noise_scalar_dict=hyper_noise_scalar_dict,
            preloads=self.preloads,
        )

    def fit_via_instance_from(
        self, instance: af.ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:
        return self.fit_via_instance_and_dataset_from(
            instance=instance,
            dataset=self.dataset,
            hyper_noise_scale=hyper_noise_scale,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be loaded after the analysis using PyAutoFit's database and
        aggregator tools.

        For this analysis the following are output:

        - The 2D charge injection dataset.
        - The clocker used for modeling / clocking CTI.
        - The settings used for modeling / clocking CTI.
        - The full 1D dataset (e.g. unmasked, used for visualizariton).

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization,and the pickled objects used by the aggregator output by this function.
        """

        self.dataset.output_to_fits(
            data_path=paths._files_path / "data.fits",
            noise_map_path=paths._files_path / "noise_map.fits",
            pre_cti_data_path=paths._files_path / "pre_cti_data.fits",
            cosmic_ray_map_path=paths._files_path / "cosmic_ray_map.fits",
            overwrite=True
        )

        paths.save_object("clocker", self.clocker)
        paths.save_object("settings_cti", self.settings_cti)
        if self.dataset_full is not None:
            paths.save_object("dataset_full", self.dataset_full)

    def visualize_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

        region_list = self.region_list_from(model=model)

        if conf.instance["visualize"]["plots"]["dataset"]["fpr_non_uniformity"]:
            region_list += ["fpr_non_uniformity"]

        visualizer.visualize_dataset(dataset=self.dataset)
        visualizer.visualize_dataset_regions(
            dataset=self.dataset, region_list=region_list
        )

        if self.dataset_full is not None:
            visualizer.visualize_dataset(
                dataset=self.dataset_full, folder_suffix="_full"
            )
            visualizer.visualize_dataset_regions(
                dataset=self.dataset_full,
                region_list=region_list,
                folder_suffix="_full",
            )

    def visualize_before_fit_combined(
        self, analyses, paths: af.DirectoryPaths, model: af.Collection
    ):
        if analyses is None:
            return

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

        region_list = self.region_list_from(model=model)

        if conf.instance["visualize"]["plots"]["dataset"]["fpr_non_uniformity"]:
            region_list += ["fpr_non_uniformity"]

        dataset_list = [analysis.dataset for analysis in analyses]

        visualizer.visualize_dataset_combined(
            dataset_list=dataset_list,
        )

        visualizer.visualize_dataset_regions_combined(
            dataset_list=dataset_list,
            region_list=region_list,
        )
        if self.dataset_full is not None:
            dataset_full_list = [analysis.dataset_full for analysis in analyses]

            visualizer.visualize_dataset_combined(
                dataset_list=dataset_full_list,
                folder_suffix="_full",
                filename_suffix="_full",
            )
            visualizer.visualize_dataset_regions_combined(
                dataset_list=dataset_full_list,
                region_list=region_list,
                folder_suffix="_full",
                filename_suffix="_full",
            )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        fit = self.fit_via_instance_from(instance=instance)
        region_list = self.region_list_from(model=instance)

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)
        visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_fit_1d_regions(
            fit=fit, during_analysis=during_analysis, region_list=region_list
        )

        if self.dataset_full is not None:
            fit_full = self.fit_via_instance_and_dataset_from(
                instance=instance, dataset=self.dataset_full
            )

            visualizer.visualize_fit(
                fit=fit_full, during_analysis=during_analysis, folder_suffix="_full"
            )
            visualizer.visualize_fit_1d_regions(
                fit=fit_full,
                during_analysis=during_analysis,
                region_list=region_list,
                folder_suffix="_full",
            )

    def visualize_combined(
        self,
        analyses: List["AnalysisImagingCI"],
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        fit_list = [
            analysis.fit_via_instance_from(instance=instance) for analysis in analyses
        ]
        region_list = self.region_list_from(model=instance)

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)
        visualizer.visualize_fit_combined(
            fit_list=fit_list, during_analysis=during_analysis
        )
        visualizer.visualize_fit_1d_regions_combined(
            fit_list=fit_list,
            region_list=region_list,
            during_analysis=during_analysis,
        )

        if self.dataset_full is not None:
            fit_list_full = [
                analysis.fit_via_instance_and_dataset_from(
                    instance=instance, dataset=analysis.dataset_full
                )
                for analysis in analyses
            ]

            visualizer.visualize_fit_combined(
                fit_list=fit_list_full,
                during_analysis=during_analysis,
                folder_suffix="_full",
            )
            visualizer.visualize_fit_1d_regions_combined(
                fit_list=fit_list_full,
                region_list=region_list,
                during_analysis=during_analysis,
                folder_suffix="_full",
            )

    def make_result(
        self,
        samples: af.SamplesPDF,
    ) -> ResultImagingCI:
        return ResultImagingCI(samples=samples, analysis=self)
