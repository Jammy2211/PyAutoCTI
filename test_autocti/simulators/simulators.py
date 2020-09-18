import autocti as ac

from test_autocti.simulators import resolution_util


def simulate__ci_uniform__parallel_x1(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "parallel_x1"

    parallel_traps = [ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)]
    parallel_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, parallel_express=2)

    for normalization in normalizations:

        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            clocker=clocker,
            pattern=pattern,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
        )


def simulate__ci_uniform__parallel_x3(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "parallel_x3"

    parallel_traps = [
        ac.TrapInstantCaptureWrap(density=0.5, release_timescale=2.0),
        ac.TrapInstantCaptureWrap(density=1.5, release_timescale=5.0),
        ac.TrapInstantCaptureWrap(density=2.5, release_timescale=20.0),
    ]

    parallel_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, parallel_express=2)

    for normalization in normalizations:
        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            pattern=pattern,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            clocker=clocker,
        )


def simulate__ci_uniform__serial_x1(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "serial_x1"

    serial_traps = [ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)]
    serial_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, serial_express=2)

    for normalization in normalizations:

        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            pattern=pattern,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            clocker=clocker,
        )


def simulate__ci_uniform__serial_x3(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "serial_x3"

    serial_traps = [
        ac.TrapInstantCaptureWrap(density=0.5, release_timescale=2.0),
        ac.TrapInstantCaptureWrap(density=1.5, release_timescale=5.0),
        ac.TrapInstantCaptureWrap(density=2.5, release_timescale=20.0),
    ]
    serial_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, serial_express=2)

    for normalization in normalizations:
        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            pattern=pattern,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            clocker=clocker,
        )


def simulate__ci_uniform__parallel_x1__serial_x1(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "parallel_x1__serial_x1"

    parallel_traps = [ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)]
    parallel_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    serial_traps = [ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)]
    serial_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, parallel_express=2, serial_express=2)

    for normalization in normalizations:

        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            pattern=pattern,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            clocker=clocker,
        )


def simulate__ci_uniform__parallel_x3__serial_x3(resolution, normalizations):

    ci_data_type = "ci_uniform"
    ci_data_model = "parallel_x3__serial_x3"

    parallel_traps = [
        ac.TrapInstantCaptureWrap(density=0.5, release_timescale=2.0),
        ac.TrapInstantCaptureWrap(density=1.5, release_timescale=5.0),
        ac.TrapInstantCaptureWrap(density=2.5, release_timescale=20.0),
    ]

    parallel_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    serial_traps = [
        ac.TrapInstantCaptureWrap(density=0.5, release_timescale=2.0),
        ac.TrapInstantCaptureWrap(density=1.5, release_timescale=5.0),
        ac.TrapInstantCaptureWrap(density=2.5, release_timescale=20.0),
    ]
    serial_ccd = ac.CCDWrap(
        well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
    )

    clocker = ac.ClockerWrap(iterations=1, parallel_express=2, serial_express=2)

    for normalization in normalizations:
        ci_regions = resolution_util.ci_regions_from_resolution(resolution=resolution)
        pattern = ac.ci.CIPatternUniform(
            normalization=normalization, regions=ci_regions
        )

        resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
            ci_data_type=ci_data_type,
            ci_data_model=ci_data_model,
            resolution=resolution,
            pattern=pattern,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            clocker=clocker,
        )


# def simulate__ci_uniform_cosmic_rays__parallel_x1(resolution, normalizations):
#
#     ci_data_type = "ci_uniform_cosmic_rays"
#     ci_data_model = "parallel_x1"
#
#     parallel_traps = ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)
#     parallel_ccd = ac.CCDWrap(
#         well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
#     )
#
#     clocker = ac.ClockerWrap(iterations=1, parallel_express=2)
#
#     shape = resolution_util.shape_2d_from_resolution(
#         resolution=resolution
#     )
#
#     for normalization in normalizations:
#
#         ci_regions = resolution_util.ci_regions_from_resolution(
#             resolution=resolution
#         )
#         pattern = ac.ci.CIPatternUniform(
#             normalization=normalization, regions=ci_regions
#         )
#
#         cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
#             shape=shape, well_depth=parallel_ccd.well_notch_depth
#         )
#
#         return resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
#             ci_data_type=ci_data_type,
#             ci_data_model=ci_data_model,
#             resolution=resolution,
#             pattern=pattern,
#             parallel_traps=parallel_traps,
#             parallel_ccd=parallel_ccd,
#             clocker=clocker,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#
# def simulate__ci_uniform_cosmic_rays__serial_x1(resolution, normalizations):
#
#     ci_data_type = "ci_uniform_cosmic_rays"
#     ci_data_model = "serial_x1"
#
#     serial_traps = ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)
#     serial_ccd = ac.CCDWrap(
#         well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
#     )
#
#     clocker = ac.ClockerWrap(iterations=1, serial_express=2)
#
#     shape = resolution_util.shape_2d_from_resolution(
#         resolution=resolution
#     )
#
#     for normalization in normalizations:
#
#         ci_regions = resolution_util.ci_regions_from_resolution(
#             resolution=resolution
#         )
#         pattern = ac.ci.CIPatternUniform(
#             normalization=normalization, regions=ci_regions
#         )
#
#         cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
#             shape=shape, well_depth=serial_ccd.well_notch_depth
#         )
#
#         resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
#             ci_data_type=ci_data_type,
#             ci_data_model=ci_data_model,
#             resolution=resolution,
#             pattern=pattern,
#             serial_traps=serial_traps,
#             serial_ccd=serial_ccd,
#             clocker=clocker,
#             cosmic_ray_map=cosmic_ray_map,
#         )
#
#
# def simulate__ci_uniform_cosmic_rays__parallel_x1__serial_x1(
#     resolution, normalizations
# ):
#
#     ci_data_type = "ci_uniform_cosmic_rays"
#     ci_data_model = "parallel_x1__serial_x1"
#
#     parallel_traps = ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)
#     parallel_ccd = ac.CCDWrap(
#         well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
#     )
#
#     serial_traps = ac.TrapInstantCaptureWrap(density=1.0, release_timescale=3.0)
#     serial_ccd = ac.CCDWrap(
#         well_fill_power=0.8, full_well_depth=8.47e4, well_notch_depth=1e-7
#     )
#
#     clocker = ac.ClockerWrap(iterations=1, parallel_express=1, serial_express=2)
#
#     shape = resolution_util.shape_2d_from_resolution(
#         resolution=resolution
#     )
#
#     for normalization in normalizations:
#
#         ci_regions = resolution_util.ci_regions_from_resolution(
#             resolution=resolution
#         )
#         pattern = ac.ci.CIPatternUniform(
#             normalization=normalization, regions=ci_regions
#         )
#
#         cosmic_ray_map = cosmic_ray_map_from_shape_and_well_depth(
#             shape=shape, well_depth=parallel_ccd.well_notch_depth
#         )
#
#         resolution_util.simulate_ci_data_from_ci_normalization_region_and_cti_model(
#             ci_data_type=ci_data_type,
#             ci_data_model=ci_data_model,
#             resolution=resolution,
#             pattern=pattern,
#             parallel_traps=parallel_traps,
#             parallel_ccd=parallel_ccd,
#             serial_traps=serial_traps,
#             serial_ccd=serial_ccd,
#             clocker=clocker,
#             cosmic_ray_map=cosmic_ray_map,
#         )


# def cosmic_ray_map_from_shape_and_well_depth(shape, well_depth):
#     # We use the LA Cosmic algorithm to simulate and add cosmic rays to our ci pre cti image. This routine randomly
#     # generates cosmimc rays based on realistic cosmic ray rates expected. These cosmic rays will then be added to our
#     # ci pre-cti image in the simulaate function below, and subject to CTI according to the CTI model.
#     cosmic_path = af.util.create_path(
#         path=workspace_path, folder_names=["scripts", "cosmic_rays"]
#     )
#
#     cosmic_ray_maker = cosmics.CosmicRays(
#         shape=shape,
#         cr_fluxscaling=1.0,
#         cr_length_file=cosmic_path + "crlength_v2.fits",
#         cr_distance_file=cosmic_path + "crdist.fits",
#         log=logger,
#     )
#     cosmic_ray_maker.set_ifiles()
#     return cosmic_ray_maker.drawEventsToCoveringFactor(limit=well_depth)
