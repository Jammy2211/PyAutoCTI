from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern
from autocti.data import util

from test.simulation import simulation_util

import os
import shutil

def simulate_ci_data_from_ci_normalization_region_and_cti_model(data_name, data_resolution, normalization, pattern,
                                                                cti_params, cti_settings, read_noise=None):

    shape = simulation_util.shape_from_data_resolution(data_resolution=data_resolution)
    ci_regions = simulation_util.ci_regions_from_data_resolution(data_resolution=data_resolution)
    frame_geometry = simulation_util.frame_geometry_from_data_resolution(data_resolution=data_resolution)

    ci_pre_cti = pattern.simulate_ci_pre_cti(shape=shape)

    data = ci_data.simulate(ci_pre_cti=ci_pre_cti, frame_geometry=frame_geometry, ci_pattern=pattern,
                            cti_settings=cti_settings, cti_params=cti_params, read_noise=read_noise)

    # Now, lets output this simulated ccd-data to the test/data folder.
    path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

    data_path = path + 'data/' + data_name

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_path += '/' + data_resolution

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    util.numpy_array_2d_to_fits(array_2d=data.image, file_path=data_path + '/ci_image_' + str(index) + '.fits'),
             sim_ci_datas, range(len(sim_ci_datas))))

    pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)