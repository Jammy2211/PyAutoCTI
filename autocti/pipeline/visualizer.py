import autofit as af
from autocti.plot import mat_objs
from autocti.plot import plotters
from autocti.plot import ci_imaging_plots, ci_fit_plots
import copy


def setting(section, name):
    return af.conf.instance.visualize_plots.get(section, name, bool)


def plot_setting(section, name):
    return setting(section, name)


class AbstractVisualizer:
    def __init__(self, image_path):

        self.plotter = plotters.Plotter(
            output=mat_objs.Output(path=image_path, format="png")
        )
        self.sub_plotter = plotters.SubPlotter(
            output=mat_objs.Output(path=image_path + "subplots/", format="png")
        )
        self.include = plotters.Include()


class PhaseDatasetVisualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.plot_subplot_dataset = plot_setting("dataset", "subplot_dataset")
        self.plot_dataset_data = plot_setting("dataset", "data")
        self.plot_dataset_noise_map = plot_setting("dataset", "noise_map")
        self.plot_dataset_signal_to_noise_map = plot_setting(
            "dataset", "signal_to_noise_map"
        )
        self.plot_dataset_cosmic_ray_map = plot_setting("dataset", "cosmic_ray_map")

        self.plot_fit_all_at_end_png = plot_setting("fit", "all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("fit", "all_at_end_fits")
        self.plot_subplot_fit = plot_setting("fit", "subplot_fit")

        self.plot_fit_data = plot_setting("fit", "data")
        self.plot_fit_noise_map = plot_setting("fit", "noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("fit", "signal_to_noise_map")
        self.plot_fit_ci_pre_cti = plot_setting("fit", "ci_pre_cti")
        self.plot_fit_ci_post_cti = plot_setting("fit", "ci_post_cti")
        self.plot_fit_residual_map = plot_setting("fit", "residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "fit", "normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("fit", "chi_squared_map")


class PhaseCIImagingVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path, results=None):
        super(PhaseCIImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.visualize_ci_imaging()

        if results is not None and results.last is not None:

            self.visualize_hyper_images(last_results=results.last)

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def visualize_ci_imaging(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "ci_imaging/"
        )

        if self.plot_subplot_dataset:
            ci_imaging_plots.subplot_ci_imaging(
                ci_imaging=self.masked_imaging.imaging,
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        ci_imaging_plots.individual(
            ci_imaging=self.masked_imaging.imaging,
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_cosmic_ray_map=self.plot_dataset_cosmic_ray_map,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_ci_imaging/"
        )

        if self.plot_subplot_fit:
            ci_fit_plots.subplot_ci_fit(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        ci_fit_plots.individuals(
            fit=fit,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_ci_pre_cti=self.plot_fit_ci_pre_cti,
            plot_ci_post_cti=self.plot_fit_ci_post_cti,
            plot_residual_map=self.plot_fit_residual_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            include=self.include,
            plotter=plotter,
        )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                ci_fit_plots.individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_ci_pre_cti=True,
                    plot_ci_post_cti=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    include=self.include,
                    plotter=plotter,
                )

            if self.plot_fit_all_at_end_fits:

                self.visualize_fit_in_fits(fit=fit)

    def visualize_fit_in_fits(self, fit):

        fits_plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/fits/", format="fits"
        )

        ci_fit_plots.individuals(
            fit=fit,
            plot_image=True,
            plot_noise_map=True,
            plot_signal_to_noise_map=True,
            plot_ci_pre_cti=True,
            plot_ci_post_cti=True,
            plot_residual_map=True,
            plot_normalized_residual_map=True,
            plot_chi_squared_map=True,
            include=self.include,
            plotter=fits_plotter,
        )
