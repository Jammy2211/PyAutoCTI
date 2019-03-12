from autofit.tools import path_util
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern

import os

test_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

def shape_from_ci_data_resolution(ci_data_resolution):

    if ci_data_resolution == 'patch':
        return (36, 36)
    elif ci_data_resolution == 'lowres':
        return (100, 100)
    elif ci_data_resolution == 'midres':
        return (300, 300)
    elif ci_data_resolution == 'highres':
        return (600, 600)
    else:
        raise ValueError('An invalid data resolution was entered - ', ci_data_resolution)

def ci_regions_from_ci_data_resolution(ci_data_resolution):

    if ci_data_resolution == 'patch':
        return [(1, 7, 3, 30), (17, 23, 3, 30)]
    elif ci_data_resolution == 'lowres':
        return [(10, 30, 10, 80), (60, 80, 10, 80)]
    elif ci_data_resolution == 'midres':
        return [(10, 40, 10, 280), (110, 140, 10, 280), (210, 240, 10, 280)]
    elif ci_data_resolution == 'highres':
        return [(10, 40, 10, 580), (110, 140, 10, 580), (210, 240, 10, 580),
                (310, 340, 10, 580), (410, 440, 10, 580), (510, 540, 10, 580)]
    else:
        raise ValueError('An invalid data resolution was entered - ', ci_data_resolution)

def frame_geometry_from_ci_data_resolution(ci_data_resolution):

    if ci_data_resolution == 'patch':
        return CIFrameGeometryIntegration.patch()
    elif ci_data_resolution == 'lowres':
        return CIFrameGeometryIntegration.low_res()
    elif ci_data_resolution == 'midres':
        return CIFrameGeometryIntegration.mid_res()
    elif ci_data_resolution == 'highres':
        return CIFrameGeometryIntegration.high_res()
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', ci_data_resolution)

def ci_data_resolution_from_shape(shape):

    if shape == (36, 36):
        return 'patch'
    elif shape == (100, 100):
        return 'lowres'
    elif shape == (300, 300):
        return 'midres'
    elif shape == (600, 600):
        return 'highres'
    else:
        raise ValueError('An invalid shape was entered when generating the data-type - ', shape)

def load_test_ci_data(ci_data_type, ci_data_model, ci_data_resolution, normalization):

    frame_geometry = frame_geometry_from_ci_data_resolution(ci_data_resolution=ci_data_resolution)
    ci_regions = ci_regions_from_ci_data_resolution(ci_data_resolution=ci_data_resolution)

    pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)
    data_path = path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=['data', ci_data_type, ci_data_model, ci_data_resolution])

    normalization = str(int(pattern.normalization))

    return ci_data.ci_data_from_fits(frame_geometry=frame_geometry, ci_pattern=pattern,
                                     image_path=data_path + 'image_' + normalization + '.fits',
                                     noise_map_path=data_path + 'noise_map_' + normalization + '.fits',
                                     ci_pre_cti_path=data_path + 'ci_pre_cti_' + normalization + '.fits')

class CIFrameGeometryIntegration(ci_frame.FrameGeometry):

    def __init__(self, corner, parallel_overscan, serial_prescan, serial_overscan):
        """This class represents the quadrant geometry of an integration quadrant."""
        super(CIFrameGeometryIntegration, self).__init__(corner=corner, parallel_overscan=parallel_overscan,
                                                         serial_prescan=serial_prescan, serial_overscan=serial_overscan)

    @classmethod
    def patch(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameGeometryIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                          serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                          serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def low_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameGeometryIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((90, 100, 10, 80)),
                                          serial_overscan=ci_frame.Region((0, 90, 80, 100)),
                                          serial_prescan=ci_frame.Region((0, 100, 0, 10)))

    @classmethod
    def mid_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameGeometryIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((280, 300, 10, 280)),
                                          serial_overscan=ci_frame.Region((0, 280, 280, 300)),
                                          serial_prescan=ci_frame.Region((0, 300, 0, 10)))

    @classmethod
    def high_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameGeometryIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((580, 600, 20, 580)),
                                          serial_overscan=ci_frame.Region((0, 580, 580, 600)),
                                          serial_prescan=ci_frame.Region((0, 600, 20, 1)))