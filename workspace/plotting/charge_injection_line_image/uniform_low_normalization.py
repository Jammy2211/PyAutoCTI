from autocti.data import util
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern
from autocti.charge_injection import ci_data

from autocti.model import arctic_settings
from autocti.model import arctic_params

from matplotlib import pyplot as plt
from autocti.charge_injection.plotters import ci_data_plotters
from autocti.data.plotters import array_plotters
from autocti.data.plotters import plotter_util

import numpy as np
import os

# This tool allows one to make simulated charge injection imaging data-sets for calibrating parallel and serial charge
# transfer inefficiency, which can be used to test example pipelines and investigate CTI modeling on data-sets where
# the 'true' answer is known.

# The 'image_type' and 'model_type' determine the directory the output data folder, e.g:

# The image will be output as '/workspace/data/image_type/model_type/ci_image.fits'.
# The noise-map will be output as '/workspace/data/image_type/model_type/ci_noise_map.fits'.
# The pre cti ci image will be output as '/workspace/data/image_type/model_type/ci_pre_cti.fits'.

shape = (2066, 2119) # The shape determines the shape of the images simulated.

# Specify the charge injection regions on the CCD, which in this case is 7 equally spaced rectangular blocks.
ci_regions = [(0, 30, 51, shape[1]-20), (330, 360, 51, shape[1]-20),
              (660, 690, 51, shape[1]-20), (990, 1020, 51, shape[1]-20),
              (1320, 1350, 51, shape[1]-20), (1650, 1680, 51, shape[1]-20)]

# The normalization of every ci image - this size of this list thus determines how many images are simulated.
normalizations=[100.0]

# The frame geometry of the image being simuated.
frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

# The CTI settings of arCTIc, which models the CCD read-out including CTI. For parallel ci data, we include 'charge
# injection mode' which accounts for the fact that every pixel is transferred over the full CCD. For serial data,
# this is omitted
parallel_cti_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=True, readout_offset=0)
serial_cti_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                               charge_injection_mode=False, readout_offset=0)
cti_settings = arctic_settings.ArcticSettings(parallel=parallel_cti_settings, serial=serial_cti_settings)

# The CTI model parameters of arCTIc, which includes each trap species density / lifetime and the CCD properties for
# parallel and serial charge transfer.
parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
parallel_ccd = arctic_params.CCD(well_notch_depth=1.0e-4, well_fill_beta=0.58, well_fill_gamma=0.0, well_fill_alpha=1.0)
serial_species = arctic_params.Species(trap_density=1.0, trap_lifetime=4.0)
serial_ccd = arctic_params.CCD(well_notch_depth=1.0e-4, well_fill_beta=0.58, well_fill_gamma=0.0, well_fill_alpha=1.0)
cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species], parallel_ccd=parallel_ccd,
                                        serial_species=[serial_species], serial_ccd=serial_ccd)

# This function creates uniform blocks of non-charge injection lines, given an input normalization, regions and
# parameters describing its non-uniform properties.
ci_patterns = ci_pattern.uniform_from_lists(normalizations=normalizations, regions=ci_regions)

# Use the simulate ci patterns to generate the pre-cti charge injection images.
ci_pre_ctis = list(map(lambda ci_pattern :
                       ci_pattern.simulate_ci_pre_cti(shape=shape),
                       ci_patterns))

# Use every ci pattern to simulate a ci image.
ci_datas = list(map(lambda ci_pre_cti, ci_pattern:
                    ci_data.simulate(ci_pre_cti=ci_pre_cti, frame_geometry=frame_geometry,
                                     ci_pattern=ci_pattern, cti_settings=cti_settings, cti_params=cti_params,
                                     read_noise=4.0, noise_seed=1),
                    ci_pre_ctis, ci_patterns))

min_value = np.min(ci_datas[0].image)
max_value = np.max(ci_datas[0].image)

import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('large')

plt.figure(figsize=(34, 25))
ci_data_plotters.plot_image(ci_data=ci_datas[0], cmap='cool', as_subplot=True, aspect='auto',
                            xlabelsize=50, ylabelsize=50, titlesize=50, xyticksize=40,
                            cb_ticksize=30)
                   #         norm='symmetric_log', linthresh=84000.0, linscale=0.001,
                   #         cb_tick_values=[min_value, 10.0, 40.0, 75.0, max_value],
                   #         cb_tick_labels=['{0:.1f}'.format(min_value), '10.0', '40.0', '75.0', '100.0'])

ax = plt.gca()
parallel_overscan = ax.add_patch(patches.Rectangle((51, 0), 2048, 20, hatch='', fill=False, color='g', linestyle='--',
                                                   linewidth=2, label='Parallel Overscan'))
serial_overscan = ax.add_patch(patches.Rectangle((shape[1]-20, 20), 20, shape[0]-20, hatch='', fill=False, color='r', linestyle='--',
                                                 linewidth=2, label='Serial Overscan'))
serial_prescan = ax.add_patch(patches.Rectangle((0, 0), 51, shape[0], hatch='', fill=False, color='k', linestyle='--',
                                                linewidth=2, label='Serial Prescan'))

plt.legend(handles=[parallel_overscan, serial_overscan, serial_prescan], fontsize=37, bbox_to_anchor=(0.99, 0.99))
# plt.show()

# Now, lets output this simulated ccd-data to the data folder. First, we'll get a relative path.
path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))
output_path = path + '/plots/'
output_filename = 'ci_image'
plt.savefig(output_path + output_filename + '.png')