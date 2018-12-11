import os

from VIS_CTI_ChargeInjection import CIData as data
from VIS_CTI_ChargeInjection import CIFrame as frame
from VIS_CTI_ChargeInjection import CIPattern as pattern

path = "/gpfs/ci_data/pdtw24/CTI".format(os.path.dirname(os.path.realpath(__file__)))


def load_ci_datas(data_name, shape, cti_geometry, normalizations, ci_regions, cr_parallel=0, cr_serial=0,
                  cr_diagonal=0):
    data_path = "{}/ci_data/{}".format(path, data_name)

    ci_patterns = pattern.create_uniform_via_lists(normalizations=normalizations, regions=ci_regions)

    images = load_images(data_path, cti_geometry, ci_patterns)

    noises = load_noises(shape, cti_geometry, ci_patterns)

    cosmic_ray_masks = load_cosmic_ray_masks(data_path, cti_geometry, ci_patterns)

    masks = setup_masks(shape, cti_geometry, ci_patterns, cosmic_ray_masks, cr_parallel, cr_serial, cr_diagonal)

    ci_pre_ctis = load_ci_pre_ctis(data_path, images, cti_geometry, ci_patterns)

    return data.CIData(images, masks, noises, ci_pre_ctis)


def load_images(data_path, cti_geometry, ci_patterns):
    return list(map(lambda ci_pattern, index:
                    data.CIImage.from_fits(path=data_path, filename='/ci_data_' + str(index), hdu=0,
                                           frame_geometry=cti_geometry, ci_pattern=ci_pattern),
                    ci_patterns, range(len(ci_patterns))))


def load_noises(shape, cti_geometry, ci_patterns):
    return list(map(lambda ci_pattern: frame.CIFrame.from_single_value(value=4.0, shape=shape,
                                                                       cti_geometry=cti_geometry,
                                                                       ci_pattern=ci_pattern), ci_patterns))


def load_cosmic_ray_masks(data_path, cti_geometry, ci_patterns):
    try:

        return list(map(lambda ci_pattern, index:
                        data.CIPreCTI.from_fits(path=data_path, filename='/cosmic_ray_mask_' + str(index), hdu=0,
                                                frame_geometry=cti_geometry, ci_pattern=ci_pattern),
                        ci_patterns, range(len(ci_patterns))))

    except FileNotFoundError:

        return None


def setup_masks(shape, cti_geometry, ci_patterns, cosmic_ray_masks, cr_parallel, cr_serial, cr_diagonal):
    if cosmic_ray_masks is None:

        return list(map(lambda ci_pattern:
                        data.CIMask.create(cti_geometry=cti_geometry, ci_pattern=ci_pattern, shape=shape,
                                           regions=[(0, 330, 0, shape[1])]),
                        ci_patterns))

    elif cosmic_ray_masks is not None:

        return list(map(lambda ci_pattern, cosmic_ray_mask:
                        data.CIMask.create(cti_geometry=cti_geometry, ci_pattern=ci_pattern, shape=shape,
                                           regions=[(0, 330, 0, shape[1])], cosmic_rays=cosmic_ray_mask,
                                           cr_parallel=cr_parallel, cr_serial=cr_serial, cr_diagonal=cr_diagonal),
                        ci_patterns, cosmic_ray_masks))


def load_ci_pre_ctis(data_path, images, cti_geometry, ci_patterns):
    try:
        return list(map(lambda ci_pattern, index:
                        data.CIPreCTI.from_fits(path=data_path, filename='/ci_pre_cti_' + str(index), hdu=0,
                                                frame_geometry=cti_geometry, ci_pattern=ci_pattern),
                        ci_patterns, range(len(ci_patterns))))

    except FileNotFoundError:

        return list(map(lambda ci_image: ci_image.create_ci_pre_cti(), images))
