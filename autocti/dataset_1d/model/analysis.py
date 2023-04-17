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

    @property
    def region_list(self) -> List:
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
        if not paths.is_complete:

            if not os.environ.get("PYAUTOFIT_TEST_MODE") == "1":

                visualizer = VisualizerDataset1D(visualize_path=paths.image_path)
                visualizer.visualize_dataset_1d(dataset_1d=self.dataset)

                if self.dataset_full is not None:

                    visualizer.visualize_dataset_1d(
                        dataset_1d=self.dataset_full, folder_suffix="_full"
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

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)

        fit = self.fit_via_instance_from(instance=instance)
        visualizer.visualize_fit_1d(fit=fit, during_analysis=during_analysis)
        visualizer.visualize_fit_1d_regions(
            fit=fit, region_list=self.region_list, during_analysis=during_analysis
        )

        if self.dataset_full is not None:
            fit = self.fit_via_instance_and_dataset_from(
                instance=instance, dataset=self.dataset_full
            )
            visualizer.visualize_fit_1d(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_1d_regions(
                fit=fit, region_list=self.region_list, during_analysis=during_analysis
            )

    def visualize_combined(
        self,
        analyses: List["AnalysisDataset1D"],
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        fit_list = [
            analysis.fit_via_instance_from(instance=instance) for analysis in analyses
        ]

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)
        visualizer.visualize_fit_1d_combined(
            fit_list=fit_list, during_analysis=during_analysis
        )
        visualizer.visualize_fit_1d_region_combined(
            fit_list=fit_list,
            region_list=self.region_list,
            during_analysis=during_analysis,
        )

        if self.dataset_full is not None:

            fit_list = [
                analysis.fit_via_instance_and_dataset_from(
                    instance=instance, dataset=analysis.dataset_full
                )
                for analysis in analyses
            ]

            visualizer.visualize_fit_1d_combined(
                fit_list=fit_list, during_analysis=during_analysis
            )
            visualizer.visualize_fit_1d_region_combined(
                fit_list=fit_list,
                region_list=self.region_list,
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
