import os

import autocti as ac
import autofit as af

test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))


def shape_2d_from_resolution(resolution):

    if resolution in "patch":
        return (36, 36)
    elif resolution in "lowres":
        return (100, 100)
    elif resolution in "midres":
        return (300, 300)
    elif resolution in "highres":
        return (600, 600)
    else:
        raise ValueError("An invalid simulator resolution was entered - ", resolution)


def ci_regions_from_resolution(resolution):

    if resolution in "patch":
        return [(1, 7, 1, 30), (17, 23, 1, 30)]
    elif resolution in "lowres":
        return [(10, 30, 10, 80), (60, 80, 10, 80)]
    elif resolution in "midres":
        return [(10, 40, 10, 280), (110, 140, 10, 280), (210, 240, 10, 280)]
    elif resolution in "highres":
        return [
            (10, 40, 10, 580),
            (110, 140, 10, 580),
            (210, 240, 10, 580),
            (310, 340, 10, 580),
            (410, 440, 10, 580),
            (510, 540, 10, 580),
        ]
    else:
        raise ValueError("An invalid simulator resolution was entered - ", resolution)


def resolution_from_shape(shape):

    if shape == (36, 36):
        return "patch"
    elif shape == (100, 100):
        return "lowres"
    elif shape == (300, 300):
        return "midres"
    elif shape == (600, 600):
        return "highres"
    else:
        raise ValueError(
            "An invalid shape was entered when generating the dataset-type - ", shape
        )


def simulate_ci_data_from_ci_normalization_region_and_cti_model(
    ci_data_type,
    ci_data_model,
    resolution,
    clocker,
    pattern,
    parallel_traps=None,
    parallel_ccd=None,
    serial_traps=None,
    serial_ccd=None,
    read_noise=1.0,
    cosmic_ray_map=None,
):

    shape = shape_2d_from_resolution(resolution=resolution)

    ci_pre_cti = pattern.simulate_ci_pre_cti(shape=shape)

    simulator = ac.ci.SimulatorCIImaging(read_noise=read_noise)

    imaging = simulator.from_image(
        clocker=clocker,
        ci_pre_cti=ci_pre_cti,
        ci_pattern=pattern,
        parallel_traps=parallel_traps,
        parallel_ccd=parallel_ccd,
        serial_traps=serial_traps,
        serial_ccd=serial_ccd,
        cosmic_ray_map=cosmic_ray_map,
    )

    # Now, lets output this simulated ccd-simulator to the test_autocti/simulator folder.
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

    ci_data_path = af.util.create_path(
        path=test_path, folders=["dataset", ci_data_type, ci_data_model, resolution]
    )

    normalization = str(int(pattern.normalization))

    imaging.output_to_fits(
        image_path=ci_data_path + "image_" + normalization + ".fits",
        noise_map_path=ci_data_path + "noise_map_" + normalization + ".fits",
        ci_pre_cti_path=ci_data_path + "ci_pre_cti_" + normalization + ".fits",
        cosmic_ray_map_path=ci_data_path + "cosmic_ray_map_" + normalization + ".fits",
        overwrite=True,
    )


def load_test_ci_data(ci_data_type, ci_data_model, resolution, normalization):

    ci_regions = ci_regions_from_resolution(resolution=resolution)

    ci_pattern = ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)

    dataset_path = af.util.create_path(
        path=test_path, folders=["dataset", ci_data_type, ci_data_model, resolution]
    )

    normalization = str(int(ci_pattern.normalization))

    if os.path.exists(f"{dataset_path}/cosmic_ray_map_{normalization}.fits"):
        cosmic_ray_map_path = f"{dataset_path}/cosmic_ray_map_{normalization}.fits"
    else:
        cosmic_ray_map_path = None

    return ac.ci.CIImaging.from_fits(
        roe_corner=(1, 0),
        ci_pattern=ci_pattern,
        pixel_scales=0.1,
        parallel_overscan=ac.Region((33, 36, 1, 30)),
        serial_overscan=ac.Region((0, 33, 30, 36)),
        serial_prescan=ac.Region((0, 36, 0, 1)),
        image_path=f"{dataset_path}/image_{normalization}.fits",
        noise_map_path=f"{dataset_path}/noise_map_{normalization}.fits",
        ci_pre_cti_path=f"{dataset_path}/ci_pre_cti_{normalization}.fits",
        cosmic_ray_map_path=cosmic_ray_map_path,
    )


# TODO : Keeping here for now so we have the overscans need to just load thm from data resolution.

# class CIFrame(ac.ci.CIFrame):
#     @classmethod
#     def patch(cls):
#         """This class represents the quadrant geometry of an integration quadrant."""
#         return cls.manual(
#             corner=(0, 0),
#             parallel_overscan=ac.Region((33, 36, 1, 30)),
#             serial_overscan=ac.Region((0, 33, 30, 36)),
#             serial_prescan=ac.Region((0, 36, 0, 1)),
#         )
#
#     @classmethod
#     def low_res(cls):
#         """This class represents the quadrant geometry of an integration quadrant."""
#         return cls.manual(
#             corner=(0, 0),
#             parallel_overscan=ac.Region((90, 100, 10, 80)),
#             serial_overscan=ac.Region((0, 90, 80, 100)),
#             serial_prescan=ac.Region((0, 100, 0, 10)),
#         )
#
#     @classmethod
#     def mid_res(cls):
#         """This class represents the quadrant geometry of an integration quadrant."""
#         return cls.manual(
#             corner=(0, 0),
#             parallel_overscan=ac.Region((280, 300, 10, 280)),
#             serial_overscan=ac.Region((0, 280, 280, 300)),
#             serial_prescan=ac.Region((0, 300, 0, 10)),
#         )
#
#     @classmethod
#     def high_res(cls):
#         """This class represents the quadrant geometry of an integration quadrant."""
#         return cls.manual(
#             corner=(0, 0),
#             parallel_overscan=ac.Region((580, 600, 20, 580)),
#             serial_overscan=ac.Region((0, 580, 580, 600)),
#             serial_prescan=ac.Region((0, 600, 20, 1)),
#         )
