import logging
from typing import List, Optional

from autoconf import conf
from autoconf.dictable import to_dict

import autoarray as aa
import autofit as af
from autoconf.dictable import output_to_json

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
        """
        Fits a CTI model to a charge injection imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit a CTI model via a `CTI2D` object to a charge injection
        imaging dataset.

        Parameters
        ----------
        dataset
            The charge injection dataset that the model is fitted to.
        clocker
            The CTI arctic clocker used by the non-linear search and model-fit.
        settings_cti
            The settings controlling aspects of the CTI model in this model-fit.
        dataset_full
            The full dataset, which is visualized separate from the `dataset` that is fitted, which for example may
            not have the FPR masked and thus enable visualization of the FPR.
        """
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
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

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

        print(fit.figure_of_merit)

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
            data=dataset.pre_cti_data,
            cti=instance.cti,
            pixel_bounce_list=instance.pixel_bounce,
            preloads=self.preloads,
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

        - The charge injection dataset (data / noise-map / pre cti data / cosmic ray map / layout / settings etc.).
        - The mask applied to the dataset.
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

        paths.save_json(
            name="clocker",
            object_dict=to_dict(self.clocker),
        )

        paths.save_json(
            name="settings_cti",
            object_dict=to_dict(self.settings_cti),
        )

        if conf.instance["visualize"]["plots"]["combined_only"]:
            return

        def output_dataset(dataset, prefix):
            paths.save_fits(
                name="data",
                hdu=dataset.data.hdu_for_output,
                prefix=prefix,
            )
            paths.save_fits(
                name="noise_map",
                hdu=dataset.noise_map.hdu_for_output,
                prefix=prefix,
            )
            paths.save_fits(
                name="pre_cti_data",
                hdu=dataset.pre_cti_data.hdu_for_output,
                prefix=prefix,
            )
            paths.save_json(
                name="layout",
                object_dict=to_dict(dataset.layout),
                prefix=prefix,
            )
            paths.save_fits(
                name="mask",
                hdu=dataset.mask.hdu_for_output,
                prefix=prefix,
            )

            if self.dataset.settings_dict is not None:
                paths.save_json(
                    name="settings_dict",
                    object_dict=self.dataset.settings_dict,
                    prefix="dataset",
                )

        output_dataset(dataset=self.dataset, prefix="dataset")

        if self.dataset_full is not None:
            output_dataset(dataset=self.dataset_full, prefix="dataset_full")

    def in_ascending_fpr_order_from(self, quantity_list, fpr_value_list):
        if not conf.instance["visualize"]["general"]["general"][
            "subplot_ascending_fpr"
        ]:
            return quantity_list

        indexes = sorted(range(len(fpr_value_list)), key=lambda k: fpr_value_list[k])

        return [quantity_list[i] for i in indexes]

    def visualize_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        if conf.instance["visualize"]["plots"]["combined_only"]:
            return

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
        fpr_value_list = [dataset.fpr_value for dataset in dataset_list]

        dataset_list = self.in_ascending_fpr_order_from(
            quantity_list=dataset_list,
            fpr_value_list=fpr_value_list,
        )

        visualizer.visualize_dataset_combined(
            dataset_list=dataset_list,
        )

        visualizer.visualize_dataset_regions_combined(
            dataset_list=dataset_list,
            region_list=region_list,
        )

        if self.dataset_full is not None:
            dataset_full_list = [analysis.dataset_full for analysis in analyses]

            dataset_full_list = self.in_ascending_fpr_order_from(
                quantity_list=dataset_full_list,
                fpr_value_list=fpr_value_list,
            )

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
        if conf.instance["visualize"]["plots"]["combined_only"]:
            return

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
        if analyses is None:
            return

        fit_list = [
            analysis.fit_via_instance_from(instance=instance) for analysis in analyses
        ]

        fpr_value_list = [fit.dataset.fpr_value for fit in fit_list]

        fit_list = self.in_ascending_fpr_order_from(
            quantity_list=fit_list,
            fpr_value_list=fpr_value_list,
        )

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
            fit_full_list = [
                analysis.fit_via_instance_and_dataset_from(
                    instance=instance, dataset=analysis.dataset_full
                )
                for analysis in analyses
            ]

            fit_full_list = self.in_ascending_fpr_order_from(
                quantity_list=fit_full_list,
                fpr_value_list=fpr_value_list,
            )

            visualizer.visualize_fit_combined(
                fit_list=fit_full_list,
                during_analysis=during_analysis,
                folder_suffix="_full",
            )
            visualizer.visualize_fit_1d_regions_combined(
                fit_list=fit_full_list,
                region_list=region_list,
                during_analysis=during_analysis,
                folder_suffix="_full",
            )

    def make_result(
        self,
        samples: af.SamplesPDF,
    ) -> ResultImagingCI:
        return ResultImagingCI(samples=samples, analysis=self)
