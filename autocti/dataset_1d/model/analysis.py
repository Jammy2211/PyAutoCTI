import os
from typing import List, Optional

import autofit as af

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.dataset_1d.fit import FitDataset1D
from autocti.dataset_1d.model.visualizer import VisualizerDataset1D
from autocti.model.result import ResultDataset
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
        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.dataset_full = dataset_full

    def region_list_from(self) -> List:
        return ["fpr", "eper"]

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

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

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be loaded after the analysis using PyAutoFit's database and
        aggregator tools.

        For this analysis the following are output:

        - The 1D dataset.
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
        paths.save_object("dataset", self.dataset)
        paths.save_object("clocker", self.clocker)
        paths.save_object("settings_cti", self.settings_cti)
        if self.dataset_full is not None:
            paths.save_object("dataset_full", self.dataset_full)

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
            fit_list = [
                analysis.fit_via_instance_and_dataset_from(
                    instance=instance, dataset=analysis.dataset_full
                )
                for analysis in analyses
            ]

            visualizer.visualize_fit_combined(
                fit_list=fit_list, during_analysis=during_analysis
            )
            visualizer.visualize_fit_region_combined(
                fit_list=fit_list,
                region_list=region_list,
                during_analysis=during_analysis,
            )

    def make_result(
        self,
        samples: af.SamplesPDF,
        model: af.Collection,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultDataset1D:
        return ResultDataset1D(samples=samples, model=model, analysis=self)
