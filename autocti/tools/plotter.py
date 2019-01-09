import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from autocti.tools import imageio


def plot_2d_images_array(images, new_figure=True, figsize=(12, 10), cmap='gray', log_norm=False,
                         cb_plot=True, cb_label='', cb_ticksize=20, cb_labelsize=20,
                         xlabel='', ylabel='', labelsize=16, ticksize=16,
                         path=None, filename=None, file_format='.png'):
    plt.figure(figsize=figsize)

    for i in range(len(images)):
        plt.subplot(4, 3, i + 1)

        plot_2d_image(image=images[i], new_figure=False, figsize=figsize, cmap=cmap, log_norm=log_norm,
                      cb_plot=cb_plot, cb_label=cb_label, cb_ticksize=cb_ticksize, cb_labelsize=cb_labelsize,
                      xlabel=xlabel, ylabel=ylabel, labelsize=labelsize, ticksize=ticksize,
                      path=path, filename=filename, file_format=file_format)

    output_plot(path, filename, file_format)
    close_figure(new_figure)


def plot_2d_image(image, new_figure=True, figsize=(12, 10), cmap='gray', log_norm=False,
                  cb_plot=True, cb_label='', cb_ticksize=20, cb_labelsize=20,
                  xlabel='', ylabel='', labelsize=16, ticksize=16,
                  path=None, filename=None, file_format='.png'):
    if log_norm is False:
        plt.imshow(image)  # , cmap=cmap)
    elif log_norm is True:
        plt.imshow(image, cmap=cmap, norm=LogNorm(vmin=0.01, vmax=1))

    setup_colorbar(cb_plot, cb_label, cb_ticksize, cb_labelsize)
    setup_labels(xlabel, ylabel, labelsize)
    setup_tickers(ticksize)
    output_plot(path, filename, file_format)
    close_figure(new_figure)


def setup_colorbar(plot_colorbar, label, ticksize, fontsize):
    if plot_colorbar is True:
        colorbar = plt.colorbar(label=label)
        axis = colorbar.ax
        axis.tick_params(labelsize=ticksize)
        text = axis.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=fontsize)
        text.set_font_properties(font)


def setup_labels(xlabel, ylabel, labelsize):
    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)


def setup_tickers(ticksize):
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)


def output_plot(path, filename, file_format):
    if path is not None and filename is not None:
        imageio.make_path_if_does_not_exist(path)
        plt.savefig(path + filename + file_format, bbox_inches='tight')
    else:
        plt.show()


def close_figure(new_figure):
    if new_figure is True:
        plt.close()
