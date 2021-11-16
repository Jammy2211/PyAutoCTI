from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.fit.plot import fit_imaging_plotters
from autocti.charge_injection.fit import FitImagingCI


class FitImagingCIPlotter(fit_imaging_plotters.FitImagingMetaPlotter):
    def __init__(
        self,
        fit: FitImagingCI,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d

    @property
    def visuals_with_include_2d(self):

        visuals_2d = super().visuals_with_include_2d

        return visuals_2d + self.visuals_2d.__class__(
            parallel_overscan=self.extract_2d(
                "parallel_overscan", self.fit.imaging_ci.layout.parallel_overscan
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.fit.imaging_ci.layout.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan", self.fit.imaging_ci.layout.serial_overscan
            ),
        )

    @property
    def extract_line_from(self):
        return self.fit.imaging_ci.layout.extract_line_from

    def figures_2d(
        self,
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        post_cti_data=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
    ):

        super().figures_2d(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

        if pre_cti_data:

            self.mat_plot_2d.plot_array(
                array=self.fit.pre_cti_data,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="CI Pre CTI Image", filename="pre_cti_data"
                ),
            )

        if post_cti_data:

            self.mat_plot_2d.plot_array(
                array=self.fit.post_cti_data,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(
                    title="CI Post CTI Image", filename="post_cti_data"
                ),
            )

    def figures_1d_ci_line_region(
        self,
        line_region,
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        post_cti_data=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
    ):

        if image:

            line = self.extract_line_from(array=self.fit.image, line_region=line_region)

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Image Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"image_{line_region}",
                ),
            )

        if noise_map:

            line = self.extract_line_from(array=self.fit.image, line_region=line_region)

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Noise-Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"noise_map_{line_region}",
                ),
            )

        if signal_to_noise_map:

            line = self.extract_line_from(
                array=self.fit.signal_to_noise_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Signal-To-Noise Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"signal_to_noise_map_{line_region}",
                ),
            )

        if pre_cti_data:

            line = self.extract_line_from(
                array=self.fit.pre_cti_data, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Pre CTI Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"pre_cti_data_{line_region}",
                ),
            )

        if post_cti_data:

            line = self.extract_line_from(
                array=self.fit.post_cti_data, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"CI Post CTI Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"post_cti_data_{line_region}",
                ),
            )

        if residual_map:

            line = self.extract_line_from(
                array=self.fit.residual_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Resdial-Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"residual_map_{line_region}",
                ),
            )

        if normalized_residual_map:

            line = self.extract_line_from(
                array=self.fit.normalized_residual_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Normalized Residual Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"normalized_residual_map_{line_region}",
                ),
            )

        if chi_squared_map:

            line = self.extract_line_from(
                array=self.fit.chi_squared_map, line_region=line_region
            )

            self.mat_plot_1d.plot_yx(
                y=line,
                x=range(len(line)),
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title=f"Chi-Squared Map Line {line_region}",
                    ylabel="Image",
                    xlabel="Pixel No.",
                    filename=f"chi_squared_map_{line_region}",
                ),
            )

    def subplot(
        self,
        image=False,
        noise_map=False,
        signal_to_noise_map=False,
        pre_cti_data=False,
        post_cti_data=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        auto_filename="subplot_fit_ci",
    ):

        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            pre_cti_data=pre_cti_data,
            post_cti_data=post_cti_data,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit_ci(self):
        return self.subplot(
            image=True,
            signal_to_noise_map=True,
            pre_cti_data=True,
            post_cti_data=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )

    def subplot_1d_ci_line_region(self, line_region):
        """
        Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.
        """

        self.open_subplot_figure(number_subplots=6)

        self.figures_1d_ci_line_region(image=True, line_region=line_region)
        self.figures_1d_ci_line_region(
            signal_to_noise_map=True, line_region=line_region
        )
        self.figures_1d_ci_line_region(pre_cti_data=True, line_region=line_region)
        self.figures_1d_ci_line_region(post_cti_data=True, line_region=line_region)
        self.figures_1d_ci_line_region(
            normalized_residual_map=True, line_region=line_region
        )
        self.figures_1d_ci_line_region(chi_squared_map=True, line_region=line_region)

        self.mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_1d_fit_ci_{line_region}"
        )
        self.close_subplot_figure()

    def subplot_noise_scaling_map_list(self):
        """Plot the observed chi_squared_map of the ccd simulator.

        Set *autocti.simulator.plotter.plotter* for a description of all input parameters not described below.

        Parameters
        -----------
        chi_squared_map : CIFrame
            The chi_squared_map of the dataset.
        """

        self.open_subplot_figure(number_subplots=len(self.fit.noise_scaling_map_list))

        for index in range(len(self.fit.noise_scaling_map_list)):

            self.mat_plot_2d.plot_array(
                array=self.fit.noise_scaling_map_list[index],
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=AutoLabels(title=f"Noise Scaling Map {index}"),
            )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"subplot_noise_scaling_map_list"
        )
        self.mat_plot_2d.figure.close()
