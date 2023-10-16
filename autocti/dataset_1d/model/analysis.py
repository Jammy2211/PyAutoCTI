from typing import List, Optional

from autoconf import conf
from autoconf.dictable import to_dict

import autofit as af
from autoconf.dictable import output_to_json

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.dataset_1d.fit import FitDataset1D
from autocti.dataset_1d.model.visualizer import VisualizerDataset1D
from autocti.dataset_1d.model.result import ResultDataset1D
from autocti.model.settings import SettingsCTI1D
from autocti.clocker.one_d import Clocker1D


class AnalysisDataset1D(af.Analysis):
    def __init__(
        self,
        dataset: Dataset1D,
        clocker: Clocker1D,
        settings_cti: SettingsCTI1D = SettingsCTI1D(),
        dataset_full: Optional[Dataset1D] = None,
    ):
        """
        Fits a CTI model to a 1D CTI dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit a CTI model via a `CTI1D` object to a charge injection
        imaging dataset.

        Parameters
        ----------
        dataset
            The 1D CTI dataset that the model is fitted to.
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

    def region_list_from(self) -> List:
        return ["fpr", "eper"]

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function:

         1) Visualizes the 1D dataset, which does not change during the analysis and thus can be done once.

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

        self.settings_cti.check_total_density_within_range(traps=instance.cti.trap_list)

        fit = self.fit_via_instance_from(instance=instance)

        return fit.log_likelihood

    def fit_via_instance_and_dataset_from(
        self, instance: af.ModelInstance, dataset: Dataset1D
    ) -> FitDataset1D:
        post_cti_data = self.clocker.add_cti(
            data=dataset.pre_cti_data, cti=instance.cti
        )

        return FitDataset1D(dataset=dataset, post_cti_data=post_cti_data)

    def fit_via_instance_from(self, instance: af.ModelInstance) -> FitDataset1D:
        return self.fit_via_instance_and_dataset_from(
            instance=instance, dataset=self.dataset
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be loaded after the analysis using PyAutoFit's database and
        aggregator tools.

        For this analysis the following are output:

        - The 1D dataset (data / noise-map / pre cti data / layout / settings etc.).
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
            paths.save_fits(
                name="mask",
                hdu=dataset.mask.hdu_for_output,
                prefix=prefix,
            )
            paths.save_json(
                name="layout",
                object_dict=to_dict(dataset.layout),
                prefix=prefix,
            )

        output_dataset(dataset=self.dataset, prefix="dataset")

        if self.dataset_full is not None:
            output_dataset(dataset=self.dataset_full, prefix="dataset_full")

        paths.save_json(
            name="clocker",
            object_dict=to_dict(self.clocker),
        )

        paths.save_json(
            name="settings_cti",
            object_dict=to_dict(self.settings_cti),
        )

    def in_ascending_fpr_order_from(self, quantity_list, fpr_value_list):
        if not conf.instance["visualize"]["general"]["general"][
            "subplot_ascending_fpr"
        ]:
            return quantity_list

        indexes = sorted(range(len(fpr_value_list)), key=lambda k: fpr_value_list[k])

        return [quantity_list[i] for i in indexes]

    def visualize_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        region_list = self.region_list_from()

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)
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

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)

        region_list = self.region_list_from()

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
                dataset_list=dataset_full_list, folder_suffix="_full"
            )
            visualizer.visualize_dataset_regions_combined(
                dataset_list=dataset_full_list,
                region_list=region_list,
                folder_suffix="_full",
            )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        region_list = self.region_list_from()

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)

        fit = self.fit_via_instance_from(instance=instance)
        visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_fit_regions(
            fit=fit, region_list=region_list, during_analysis=during_analysis
        )

        if self.dataset_full is not None:
            fit = self.fit_via_instance_and_dataset_from(
                instance=instance, dataset=self.dataset_full
            )
            visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_regions(
                fit=fit, region_list=region_list, during_analysis=during_analysis
            )

    def visualize_combined(
        self,
        analyses: List["AnalysisDataset1D"],
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

        region_list = self.region_list_from()

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)
        visualizer.visualize_fit_combined(
            fit_list=fit_list, during_analysis=during_analysis
        )
        visualizer.visualize_fit_region_combined(
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
                fit_list=fit_full_list, during_analysis=during_analysis
            )
            visualizer.visualize_fit_region_combined(
                fit_list=fit_full_list,
                region_list=region_list,
                during_analysis=during_analysis,
            )

    def make_result(
        self,
        samples: af.SamplesPDF,
    ) -> ResultDataset1D:
        return ResultDataset1D(samples=samples, analysis=self)
