from autofit.tools import path_util
from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern
from autocti.model import arctic_params
from autocti.model import arctic_settings

from test.simulation import simulation_util

import os

def simulate_ci_data_from_ci_normalization_region_and_cti_model(ci_data_type, ci_data_model, ci_data_resolution,
                                                                pattern, cti_params, cti_settings, read_noise=1.0):

    shape = simulation_util.shape_from_ci_data_resolution(ci_data_resolution=ci_data_resolution)
    frame_geometry = simulation_util.frame_geometry_from_ci_data_resolution(ci_data_resolution=ci_data_resolution)

    ci_pre_cti = pattern.simulate_ci_pre_cti(shape=shape)

    data = ci_data.simulate(ci_pre_cti=ci_pre_cti, frame_geometry=frame_geometry, ci_pattern=pattern,
                            cti_settings=cti_settings, cti_params=cti_params, read_noise=read_noise)

    # Now, lets output this simulated ccd-data to the test/data folder.
    test_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

    ci_data_path = path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=['data', ci_data_type, ci_data_model, ci_data_resolution])

    normalization = str(int(pattern.normalization))

    ci_data.output_ci_data_to_fits(ci_data=data,
                                   image_path=ci_data_path + 'ci_image_' + normalization + '.fits',
                                   noise_map_path=ci_data_path + 'ci_noise_map_'  + normalization + '.fits',
                                   ci_pre_cti_path=ci_data_path + 'ci_pre_cti_'  + normalization + '.fits',
                                   overwrite=True)

def make_ci_uniform_parallel_x1_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'parallel_x1_species'

    parallel_species = arctic_params.Species(trap_density=1.0, trap_lifetime=3.0)
    parallel_ccd = arctic_params.CCD(well_notch_depth=0.0, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)
    cti_params = arctic_params.ArcticParams(parallel_ccd=parallel_ccd, parallel_species=[parallel_species])

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)
            
def make_ci_uniform_serial_x1_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'serial_x1_species'

    serial_species = arctic_params.Species(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.00, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)
    cti_params = arctic_params.ArcticParams(serial_ccd=serial_ccd, serial_species=[serial_species])

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)

def make_ci_uniform_parallel_and_serial_x1_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'parallel_and_serial_x1_species'

    parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.5)
    parallel_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)

    serial_species = arctic_params.Species(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.00, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(parallel_ccd=parallel_ccd, parallel_species=[parallel_species],
                                            serial_ccd=serial_ccd, serial_species=[serial_species])

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)

    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)


def make_ci_uniform_parallel_x3_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'parallel_x3_species'

    parallel_species_0 = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)
    parallel_species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=5.0)
    parallel_species_2 = arctic_params.Species(trap_density=2.5, trap_lifetime=20.0)
    parallel_ccd = arctic_params.CCD(well_notch_depth=0.0, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)
    cti_params = arctic_params.ArcticParams(parallel_ccd=parallel_ccd,
                                parallel_species=[parallel_species_0, parallel_species_1, parallel_species_2])

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)

def make_ci_uniform_serial_x3_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'serial_x3_species'

    serial_species_0 = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)
    serial_species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=5.0)
    serial_species_2 = arctic_params.Species(trap_density=2.5, trap_lifetime=20.0)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.00, well_fill_alpha=1.0,
                                   well_fill_beta=0.5, well_fill_gamma=0.0)
    cti_params = arctic_params.ArcticParams(serial_ccd=serial_ccd,
                                            serial_species=[serial_species_0, serial_species_1, serial_species_2])

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                               charge_injection_mode=False, readout_offset=0)
    cti_settings = arctic_settings.ArcticSettings(serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)


def make_ci_uniform_parallel_and_serial_x3_species(data_resolutions, normalizations):

    ci_data_type = 'ci_uniform'
    ci_data_model = 'parallel_and_serial_x3_species'

    parallel_species_0 = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)
    parallel_species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=5.0)
    parallel_species_2 = arctic_params.Species(trap_density=2.5, trap_lifetime=20.0)
    parallel_ccd = arctic_params.CCD(well_notch_depth=0.01, well_fill_alpha=1.0,
                                     well_fill_beta=0.5, well_fill_gamma=0.0)

    serial_species_0 = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)
    serial_species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=5.0)
    serial_species_2 = arctic_params.Species(trap_density=2.5, trap_lifetime=20.0)
    serial_ccd = arctic_params.CCD(well_notch_depth=0.00, well_fill_alpha=1.0,
                                   well_fill_beta=0.5, well_fill_gamma=0.0)

    cti_params = arctic_params.ArcticParams(parallel_ccd=parallel_ccd,
                                            parallel_species=[parallel_species_0, parallel_species_1,
                                                              parallel_species_2],
                                            serial_ccd=serial_ccd,
                                            serial_species=[serial_species_0, serial_species_1, serial_species_2])

    parallel_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)

    serial_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                               charge_injection_mode=False, readout_offset=0)

    cti_settings = arctic_settings.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulation_util.ci_regions_from_ci_data_resolution(ci_data_resolution=data_resolution)
            pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type, ci_data_model=ci_data_model, ci_data_resolution=data_resolution,
                pattern=pattern, cti_params=cti_params, cti_settings=cti_settings)