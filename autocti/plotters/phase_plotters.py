from matplotlib import pyplot as plt

from autocti.plotters import ci_plotter_util, ci_imaging_plotters
from autoarray.util import array_util


def plot_ci_data_for_phase(
    ci_datas_extracted,
    extract_array_from_mask,
    plot_as_subplot,
    plot_image,
    plot_noise_map,
    plot_ci_pre_cti,
    plot_signal_to_noise_map,
    plot_parallel_front_edge_line,
    plot_parallel_trails_line,
    plot_serial_front_edge_line,
    plot_serial_trails_line,
    visualize_path,
):

    plot_ci_data_arrays_for_phase(
        ci_datas_extracted=ci_datas_extracted,
        plot_as_subplot=plot_as_subplot,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_ci_pre_cti=plot_ci_pre_cti,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        visualize_path=visualize_path,
    )

    plot_ci_data_lines_for_phase(
        ci_datas_extracted=ci_datas_extracted,
        plot_as_subplot=plot_as_subplot,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_ci_pre_cti=plot_ci_pre_cti,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_parallel_front_edge_line=plot_parallel_front_edge_line,
        plot_parallel_trails_line=plot_parallel_trails_line,
        plot_serial_front_edge_line=plot_serial_front_edge_line,
        plot_serial_trails_line=plot_serial_trails_line,
        visualize_path=visualize_path,
    )


def plot_ci_data_arrays_for_phase(
    ci_datas_extracted,
    extract_array_from_mask,
    plot_as_subplot,
    plot_image,
    plot_noise_map,
    plot_ci_pre_cti,
    plot_signal_to_noise_map,
    visualize_path,
):

    for data_index in range(len(ci_datas_extracted)):

        normalization = ci_datas_extracted[data_index].ci_pattern.normalization
        output_path = (
            visualize_path
            + "/"
            + "ci_image_"
            + str(int(normalization))
            + "/structures/"
        )
        array_util.make_path_if_does_not_exist(path=output_path + "fits/")

        if plot_as_subplot:

            plot_ci_subplot(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                output_path=output_path,
                output_format="png",
            )

        plot_ci_data_individual(
            ci_data=ci_datas_extracted[data_index],
            mask=ci_datas_extracted[data_index].mask,
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_ci_pre_cti=plot_ci_pre_cti,
            plot_signal_to_noise_map=plot_signal_to_noise_map,
            output_path=output_path,
            output_format="png",
        )


def plot_ci_data_lines_for_phase(
    ci_datas_extracted,
    plot_as_subplot,
    plot_image,
    plot_noise_map,
    plot_ci_pre_cti,
    plot_signal_to_noise_map,
    plot_parallel_front_edge_line,
    plot_parallel_trails_line,
    plot_serial_front_edge_line,
    plot_serial_trails_line,
    visualize_path,
):

    line_regions = ci_plotter_util.line_regions_from_plots(
        plot_parallel_front_edge_line=plot_parallel_front_edge_line,
        plot_parallel_trails_line=plot_parallel_trails_line,
        plot_serial_front_edge_line=plot_serial_front_edge_line,
        plot_serial_trails_line=plot_serial_trails_line,
    )

    for data_index in range(len(ci_datas_extracted)):

        for line_region in line_regions:

            normalization = ci_datas_extracted[data_index].ci_pattern.normalization
            output_path = (
                visualize_path
                + "/"
                + "ci_image_"
                + str(int(normalization))
                + "/"
                + line_region
                + "/"
            )
            array_util.make_path_if_does_not_exist(path=output_path + "fits/")

            if plot_as_subplot:

                plot_ci_line_subplot(
                    ci_data=ci_datas_extracted[data_index],
                    line_region=line_region,
                    output_path=output_path,
                    output_format="png",
                )

            plot_ci_data_line_individual(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                line_region=line_region,
                plot_image=plot_image,
                plot_noise_map=plot_noise_map,
                plot_ci_pre_cti=plot_ci_pre_cti,
                plot_signal_to_noise_map=plot_signal_to_noise_map,
                output_path=output_path,
                output_format="png",
            )

            plot_ci_data_line_individual(
                ci_data=ci_datas_extracted[data_index],
                mask=ci_datas_extracted[data_index].mask,
                line_region=line_region,
                plot_image=plot_image,
                plot_noise_map=plot_noise_map,
                plot_ci_pre_cti=plot_ci_pre_cti,
                plot_signal_to_noise_map=plot_signal_to_noise_map,
                output_path=output_path,
                output_format="fits",
            )
