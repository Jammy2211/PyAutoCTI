import autofit as af

from autocti.dataset_1d.model.plotter_interface import PlotterInterfaceDataset1D


class VisualizerDataset1D(af.Visualizer):
    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        region_list = analysis.region_list_from()

        visualizer = PlotterInterfaceDataset1D(image_path=paths.image_path)
        visualizer.dataset(dataset=analysis.dataset)
        visualizer.dataset_regions(dataset=analysis.dataset, region_list=region_list)

        if analysis.dataset_full is not None:
            visualizer.dataset(dataset=analysis.dataset_full, folder_suffix="_full")
            visualizer.dataset_regions(
                dataset=analysis.dataset_full,
                region_list=region_list,
                folder_suffix="_full",
            )

    @staticmethod
    def visualize_before_fit_combined(
        analyses,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        if analyses is None:
            return

        visualizer = PlotterInterfaceDataset1D(image_path=paths.image_path)

        region_list = analyses[0].region_list_from()

        dataset_list = [analysis.dataset for analysis in analyses]
        fpr_value_list = [dataset.fpr_value for dataset in dataset_list]

        dataset_list = analyses[0].in_ascending_fpr_order_from(
            quantity_list=dataset_list,
            fpr_value_list=fpr_value_list,
        )

        visualizer.dataset_combined(
            dataset_list=dataset_list,
        )
        visualizer.dataset_regions_combined(
            dataset_list=dataset_list,
            region_list=region_list,
        )

        if analyses[0].dataset_full is not None:
            dataset_full_list = [analysis.dataset_full for analysis in analyses]

            dataset_full_list = analyses[0].in_ascending_fpr_order_from(
                quantity_list=dataset_full_list,
                fpr_value_list=fpr_value_list,
            )

            visualizer.dataset_combined(
                dataset_list=dataset_full_list, folder_suffix="_full"
            )
            visualizer.dataset_regions_combined(
                dataset_list=dataset_full_list,
                region_list=region_list,
                folder_suffix="_full",
            )

    @staticmethod
    def visualize(
        analysis,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Output images of the maximum log likelihood model inferred by the model-fit. This function is called throughout
        the non-linear search at regular intervals, and therefore provides on-the-fly visualization of how well the
        model-fit is going.

        The images output by this function are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        region_list = analysis.region_list_from()

        visualizer = PlotterInterfaceDataset1D(image_path=paths.image_path)

        fit = analysis.fit_via_instance_from(instance=instance)
        visualizer.fit(fit=fit, during_analysis=during_analysis)
        visualizer.fit_regions(
            fit=fit, region_list=region_list, during_analysis=during_analysis
        )

        if analysis.dataset_full is not None:
            fit = analysis.fit_via_instance_and_dataset_from(
                instance=instance, dataset=analysis.dataset_full
            )
            visualizer.fit(fit=fit, during_analysis=during_analysis)
            visualizer.fit_regions(
                fit=fit, region_list=region_list, during_analysis=during_analysis
            )

    @staticmethod
    def visualize_combined(
        analyses,
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

        fit_list = analyses[0].in_ascending_fpr_order_from(
            quantity_list=fit_list,
            fpr_value_list=fpr_value_list,
        )

        region_list = analyses[0].region_list_from()

        visualizer = PlotterInterfaceDataset1D(image_path=paths.image_path)
        visualizer.fit_combined(fit_list=fit_list, during_analysis=during_analysis)
        visualizer.fit_region_combined(
            fit_list=fit_list,
            region_list=region_list,
            during_analysis=during_analysis,
        )

        if analyses[0].dataset_full is not None:
            fit_full_list = [
                analysis.fit_via_instance_and_dataset_from(
                    instance=instance, dataset=analysis.dataset_full
                )
                for analysis in analyses
            ]

            fit_full_list = analyses[0].in_ascending_fpr_order_from(
                quantity_list=fit_full_list,
                fpr_value_list=fpr_value_list,
            )

            visualizer.fit_combined(
                fit_list=fit_full_list, during_analysis=during_analysis
            )
            visualizer.fit_region_combined(
                fit_list=fit_full_list,
                region_list=region_list,
                during_analysis=during_analysis,
            )
