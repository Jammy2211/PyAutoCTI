from autocti.charge_injection import ci_frame
from autocti.tools import imageio
import matplotlib.pyplot as plt
import numpy as np

def ci_regions_binned_across_serial(ci_frame, mask, path, filename, file_format='.png', line0=False):
    """Make a plot of the charge injection regions, binned across the serial direction of the CCD."""

    ci_region_rows = ci_frame.ci_pattern.regions[0].total_rows
    ci_regions = ci_frame.parallel_front_edge_arrays_from_frame(rows=(0, ci_region_rows))
    ci_regions_masks = mask.parallel_front_edge_arrays_from_frame(rows=(0, ci_region_rows))
    ci_regions_binned_arrays = list(map(lambda region, mask : ci_frame.bin_array_across_serial(region, mask),
                                        ci_regions, ci_regions_masks))

    plot_mean_of_all_binned_arrays(ci_regions_binned_arrays, path, filename, file_format, line0)
 #   sub_plots_of_each_binned_array(ci_regions_binned_arrays, path, filename, file_format, line0)

def parallel_trails_binned_across_serial(ci_frame, mask, path, filename, file_format='.png', line0=False):
    """Make a plot of the parallel trails following a charge injection region, binned across the serial direction of \
     the CCD."""

    parallel_trails_rows = ci_frame.ci_pattern.regions[1].y0 - ci_frame.ci_pattern.regions[0].y1
    parallel_trails = ci_frame.parallel_trails_arrays_from_frame(rows=(0, parallel_trails_rows))
    parallel_trails_masks = mask.parallel_trails_arrays_from_frame(rows=(0, parallel_trails_rows))
    parallel_trails_binned_arrays = list(map(lambda region, mask: ci_frame.bin_array_across_serial(region, mask),
                                             parallel_trails, parallel_trails_masks))

    plot_mean_of_all_binned_arrays(parallel_trails_binned_arrays, path, filename, file_format, line0)
 #   sub_plots_of_each_binned_array(parallel_trails_binned_arrays, path, filename, file_format, line0)

def ci_regions_binned_across_parallel(ci_frame, mask, path, filename, file_format='.png', line0=False):
    """Make a plot of the charge injection regions, binned across the serial direction of the CCD."""

    ci_region_columns = ci_frame.ci_pattern.regions[0].total_columns
    ci_regions = ci_frame.serial_front_edge_arrays_from_frame(columns=(0, ci_region_columns))
    ci_regions_masks = mask.serial_front_edge_arrays_from_frame(columns=(0, ci_region_columns))
    ci_regions_binned_arrays = list(map(lambda region, mask : ci_frame.bin_array_across_parallel(region, mask),
                                        ci_regions, ci_regions_masks))

    plot_mean_of_all_binned_arrays(ci_regions_binned_arrays, path, filename, file_format, line0)
 #   sub_plots_of_each_binned_array(ci_regions_binned_arrays, path, filename, file_format, line0)

def serial_trails_binned_across_parallel(ci_frame, mask, path, filename, file_format='.png', line0=False):
    """Make a plot of the parallel trails following a charge injection region, binned across the serial direction of \
     the CCD."""

    serial_trails_columns = ci_frame.cti_geometry.serial_overscan.total_columns
    serial_trails = ci_frame.serial_trails_arrays_from_frame(columns=(0, serial_trails_columns))
    serial_trails_masks = mask.serial_trails_arrays_from_frame(columns=(0, serial_trails_columns))
    serial_trails_binned_arrays = list(map(lambda region, mask: ci_frame.bin_array_across_parallel(region, mask),
                                           serial_trails, serial_trails_masks))

    plot_mean_of_all_binned_arrays(serial_trails_binned_arrays, path, filename, file_format, line0)
 #   sub_plots_of_each_binned_array(parallel_trails_binned_arrays, path, filename, file_format, line0)

def plot_mean_of_all_binned_arrays(arrays, path, filename, file_format, line0):

    imageio.make_path_if_does_not_exist(path)

    binned = np.mean(arrays, axis=0)
    xmax = binned.shape[0]-1

    plt.figure(figsize=(36, 20))
    plt.plot(binned)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xlabel('pixels', fontsize=36); plt.ylabel('value', fontsize=36)
    plt.xlim([0, xmax])
    if line0: plt.hlines(y=0.0, xmin=0, xmax=xmax, linestyles='dashed')
    plt.savefig(path + filename + '_binned' + file_format)
    plt.close()

def sub_plots_of_each_binned_array(arrays, path, filename, file_format, line0):
    """Make a sub-plot of each set of binned arrays."""

    imageio.make_path_if_does_not_exist(path)

    sub_plot_shape = (len(arrays), 1)
    fig, axes = plt.subplots(sub_plot_shape[0], sub_plot_shape[1], figsize=(36,20))

    xmax = arrays[0].shape[0]-1

    for i, ax in enumerate(axes.flatten()):  # flatten in case you have a second row at some point
        ax.plot(arrays[i])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim([0, xmax])
        if line0: ax.hlines(y=0.0, xmin=0, xmax=xmax, linestyles='dashed')
        ax.set_aspect('auto')

    plt.savefig(path + filename + file_format, bbox_inches='tight')
    plt.close()

def plot_ci_regions_of_ci_frame(ci_frame, path, filename, file_format='.png'):
    """Plot the charge injection regions of a CIFrame

    Parameters
    -----------
    ci_frame : VIS_CTI_ChargeInjection.ci_frame.CIFrame
        The charge injection ci_frame.
    path : str
        The path where the image is plotted.
    filename : str
        The filename of the output image.
    """

    imageio.make_path_if_does_not_exist(path)

    ci_region_rows = ci_frame.ci_pattern.regions[0].total_rows
    ci_regions = ci_frame.parallel_front_edge_arrays_from_frame(rows=(0, ci_region_rows))
    sub_plot_shape = (len(ci_regions), 1)

    fig, axes = plt.subplots(sub_plot_shape[0], sub_plot_shape[1], figsize=(36,20))

    for i, ax in enumerate(axes.flatten()):  # flatten in case you have a second row at some point
        img = ax.imshow(ci_regions[i], interpolation='nearest')
        ax.set_aspect('auto')
        plt.colorbar(img, ax=ax, orientation='horizontal', aspect=150)

    plt.savefig(path + filename + file_format, bbox_inches='tight')
    plt.close()

def plot_parallel_trails_of_ci_frame(ci_frame, path, filename, file_format='.png'):
    """Plot the charge injection regions of a CIFrame

    Parameters
    -----------
    ci_frame : VIS_CTI_ChargeInjection.ci_frame.CIFrame
        The charge injection ci_frame.
    path : str
        The path where the image is plotted.
    filename : str
        The filename of the output image.
    """

    imageio.make_path_if_does_not_exist(path)

    parallel_trails_rows = ci_frame.ci_pattern.regions[1].y0 - ci_frame.ci_pattern.regions[0].y1
    ci_regions = ci_frame.parallel_trails_arrays_from_frame(rows=(0, parallel_trails_rows // 2))
    sub_plot_shape = (len(ci_regions), 1)

    fig, axes = plt.subplots(sub_plot_shape[0], sub_plot_shape[1], figsize=(36,20))

    for i, ax in enumerate(axes.flatten()):  # flatten in case you have a second row at some point
        img = ax.imshow(ci_regions[i], interpolation='nearest')
        ax.set_aspect('auto')
        plt.colorbar(img, ax=ax, orientation='horizontal', aspect=150)

    plt.savefig(path + filename + file_format, bbox_inches='tight')
    plt.close()

  #  plot_2d_images_array(images=ci_regions, path=path, filename=filename, sub_plot_shape=sub_plot_shape)
#
# def plot_2d_images_array(images, path, filename, sub_plot_shape, file_format='.png'):
#
#
#         plot_2d_image(image=images[i], path=path, filename=filename, file_format='.png', new_figure=False)
#
#     plt.savefig(path + filename + file_format, bbox_inches='tight')
#     plt.close()
#     stop

def plot_2d_image(image, path, filename, file_format='.png', new_figure=True):

    imageio.make_path_if_does_not_exist(path)

    plt.imshow(image)
    plt.colorbar()
#    axis = colorbar.ax
#    axis.tick_params(labelsize=ticksize)
#    text = axis.yaxis.label
#    font = matplotlib.font_manager.FontProperties(size=fontsize)
#    text.set_font_properties(font)

    if new_figure is True:
        plt.savefig(path + filename + file_format, bbox_inches='tight')
        plt.close()


# # def sub_plot_shape(total):
# #
# #     if total == 1:
# #         shape = (1,1)
# #     elif total == 2:
# #         shape = (1,2)
# #     elif total == 3:
# #         shape = (1, 3)
# #     elif total == 4:
# #         shape = (2, 2)
# #     elif total == 5 or total == 6:
# #         shape = (2, 3)
# #     elif total == 7 or total == 8 or total == 9:
# #         shape = (3,3)
# #     elif total > 9 and total <= 12:
# #         shape = (3, 4)
# #     else:
# #         raise Exc.CIPlotterException('No sub-plot shape available for more than 12 charge injection images')
# #
# #     return shape
#
# #    setup_colorbar(cb_plot, cb_label, cb_ticksize, cb_labelsize)
# #    setup_labels(xlabel, ylabel, labelsize)
# #    setup_tickers(ticksize)
#
# def setup_labels(xlabel, ylabel, labelsize):
#     plt.xlabel(xlabel, fontsize=labelsize)
#     plt.ylabel(ylabel, fontsize=labelsize)
#
# def setup_tickers(ticksize):
#     plt.xticks(fontsize=ticksize)
#     plt.yticks(fontsize=ticksize)
