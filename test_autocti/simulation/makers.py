import autofit as af
import autocti as ac

from test import simulate_util

from autolens_workspace_jam.scripts.cosmic_rays import cosmics

import logging

logger = logging.getLogger(__name__)

import os

workspace_path = "{}/../../autolens_workspace/".format(
    os.path.dirname(os.path.realpath(__file__))
)


def simulate_ci_data_from_ci_normalization_region_and_cti_model(
    ci_data_type,
    ci_data_model,
    ci_data_resolution,
    pattern,
    cti_params,
    cti_settings,
    read_noise=1.0,
    cosmic_ray_map=None,
):

    shape = simulate_util.shape_from_ci_data_resolution(
        ci_data_resolution=ci_data_resolution
    )
    frame_geometry = simulate_util.frame_geometry_from_ci_data_resolution(
        ci_data_resolution=ci_data_resolution
    )

    ci_pre_cti = pattern.simulate_ci_pre_cti(shape=shape)

    data = ac.CIImaging.simulate(
        ci_pre_cti=ci_pre_cti,
        frame_geometry=frame_geometry,
        ci_pattern=pattern,
        cti_settings=cti_settings,
        cti_params=cti_params,
        read_noise=read_noise,
        cosmic_ray_map=cosmic_ray_map,
    )

    # Now, lets output this simulated ccd-simulator to the test_autoarray/simulator folder.
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

    ci_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path,
        folder_names=["dataset", ci_data_type, ci_data_model, ci_data_resolution],
    )

    normalization = str(int(pattern.normalization))

    ac.output_ci_data_to_fits(
        ci_data=data,
        image_path=ci_data_path + "image_" + normalization + ".fits",
        noise_map_path=ci_data_path + "noise_map_" + normalization + ".fits",
        ci_pre_cti_path=ci_data_path + "ci_pre_cti_" + normalization + ".fits",
        cosmic_ray_map_path=ci_data_path + "cosmic_ray_map_" + normalization + ".fits",
        overwrite=True,
    )


def make_ci_uniform_parallel_x1_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "parallel_x1"

    parallel_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.0,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume, parallel_traps=[parallel_traps]
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(parallel=parallel_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_serial_x1_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "serial_x1"

    serial_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        serial_ccd_volume=serial_ccd_volume, serial_traps=[serial_traps]
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_parallel_x1__serial_x1_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "parallel_x1__serial_x1"

    parallel_traps = ac.Trap(trap_density=0.1, trap_lifetime=1.5)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.01,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    serial_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume,
        parallel_traps=[parallel_traps],
        serial_ccd_volume=serial_ccd_volume,
        serial_traps=[serial_traps],
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    cti_settings = ac.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_parallel_x3_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "parallel_x3"

    parallel_traps_0 = ac.Trap(trap_density=0.5, trap_lifetime=2.0)
    parallel_traps_1 = ac.Trap(trap_density=1.5, trap_lifetime=5.0)
    parallel_traps_2 = ac.Trap(trap_density=2.5, trap_lifetime=20.0)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.0,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume,
        parallel_traps=[parallel_traps_0, parallel_traps_1, parallel_traps_2],
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(parallel=parallel_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_serial_x3_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "serial_x3"

    serial_traps_0 = ac.Trap(trap_density=0.5, trap_lifetime=2.0)
    serial_traps_1 = ac.Trap(trap_density=1.5, trap_lifetime=5.0)
    serial_traps_2 = ac.Trap(trap_density=2.5, trap_lifetime=20.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        serial_ccd_volume=serial_ccd_volume,
        serial_traps=[serial_traps_0, serial_traps_1, serial_traps_2],
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_parallel_x3__serial_x3_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform"
    ci_data_model = "parallel_x3__serial_x3"

    parallel_traps_0 = ac.Trap(trap_density=0.5, trap_lifetime=2.0)
    parallel_traps_1 = ac.Trap(trap_density=1.5, trap_lifetime=5.0)
    parallel_traps_2 = ac.Trap(trap_density=2.5, trap_lifetime=20.0)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.01,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    serial_traps_0 = ac.Trap(trap_density=0.5, trap_lifetime=2.0)
    serial_traps_1 = ac.Trap(trap_density=1.5, trap_lifetime=5.0)
    serial_traps_2 = ac.Trap(trap_density=2.5, trap_lifetime=20.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume,
        parallel_traps=[parallel_traps_0, parallel_traps_1, parallel_traps_2],
        serial_ccd_volume=serial_ccd_volume,
        serial_traps=[serial_traps_0, serial_traps_1, serial_traps_2],
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    cti_settings = ac.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    for data_resolution in data_resolutions:
        for normalization in normalizations:
            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
            )


def make_ci_uniform_cosmic_rays_parallel_x1_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform__cosmic_rays"
    ci_data_model = "parallel_x1"

    parallel_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.0,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume, parallel_traps=[parallel_traps]
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(parallel=parallel_settings)

    for data_resolution in data_resolutions:

        shape = simulate_util.shape_from_ci_data_resolution(
            ci_data_resolution=data_resolution
        )

        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
                shape=shape, well_depth=cti_settings.parallel.well_depth
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
                cosmic_ray_map=cosmic_ray_map,
            )


def make_ci_uniform_cosmic_rays_serial_x1_species(data_resolutions, normalizations):

    ci_data_type = "ci__uniform__cosmic_rays"
    ci_data_model = "serial_x1"

    serial_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )
    cti_params = ac.ArcticParams(
        serial_ccd_volume=serial_ccd_volume, serial_traps=[serial_traps]
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )
    cti_settings = ac.ArcticSettings(serial=serial_settings)

    for data_resolution in data_resolutions:

        shape = simulate_util.shape_from_ci_data_resolution(
            ci_data_resolution=data_resolution
        )

        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
                shape=shape, well_depth=cti_settings.serial.well_depth
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
                cosmic_ray_map=cosmic_ray_map,
            )


def make_ci_uniform_cosmic_rays_parallel_x1__serial_x1_species(
    data_resolutions, normalizations
):

    ci_data_type = "ci__uniform__cosmic_rays"
    ci_data_model = "parallel_x1__serial_x1"

    parallel_traps = ac.Trap(trap_density=0.1, trap_lifetime=1.5)
    parallel_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.01,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    serial_traps = ac.Trap(trap_density=1.0, trap_lifetime=3.0)
    serial_ccd_volume = ac.CCDVolume(
        well_notch_depth=0.00,
        well_fill_alpha=1.0,
        well_fill_beta=0.5,
        well_fill_gamma=0.0,
    )

    cti_params = ac.ArcticParams(
        parallel_ccd_volume=parallel_ccd_volume,
        parallel_traps=[parallel_traps],
        serial_ccd_volume=serial_ccd_volume,
        serial_traps=[serial_traps],
    )

    parallel_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    serial_settings = ac.Settings(
        well_depth=84700,
        niter=1,
        express=2,
        n_levels=2000,
        charge_injection_mode=False,
        readout_offset=0,
    )

    cti_settings = ac.ArcticSettings(parallel=parallel_settings, serial=serial_settings)

    for data_resolution in data_resolutions:

        shape = simulate_util.shape_from_ci_data_resolution(
            ci_data_resolution=data_resolution
        )

        for normalization in normalizations:

            ci_regions = simulate_util.ci_regions_from_ci_data_resolution(
                ci_data_resolution=data_resolution
            )
            pattern = ac.CIPatternUniform(
                normalization=normalization, regions=ci_regions
            )

            cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
                shape=shape, well_depth=cti_settings.serial.well_depth
            )

            simulate_ci_data_from_ci_normalization_region_and_cti_model(
                ci_data_type=ci_data_type,
                ci_data_model=ci_data_model,
                ci_data_resolution=data_resolution,
                pattern=pattern,
                cti_params=cti_params,
                cti_settings=cti_settings,
                cosmic_ray_map=cosmic_ray_map,
            )


def cosmic_ray_map_from_shape_and_well_depth(shape, well_depth):
    # We use the LA Cosmic algorithm to simulate and add cosmic rays to our ci pre cti image. This routine randomly
    # generates cosmimc rays based on realistic cosmic ray rates expected. These cosmic rays will then be added to our
    # ci pre-cti image in the simulaate function below, and subject to CTI according to the CTI model.
    cosmic_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=workspace_path, folder_names=["scripts", "cosmic_rays"]
    )

    cosmic_ray_maker = cosmics.CosmicRays(
        shape=shape,
        cr_fluxscaling=1.0,
        cr_length_file=cosmic_path + "crlength_v2.fits",
        cr_distance_file=cosmic_path + "crdist.fits",
        log=logger,
    )
    cosmic_ray_maker.set_ifiles()
    return cosmic_ray_maker.drawEventsToCoveringFactor(limit=well_depth)
