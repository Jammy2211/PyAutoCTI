from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern

import os

path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

def shape_from_data_resolution(data_resolution):

    if data_resolution == 'patch':
        return (36, 36)
    elif data_resolution == 'lowres':
        return (120, 120)
    elif data_resolution == 'midres':
        return (300, 300)
    elif data_resolution == 'highres':
        return (600, 600)
    else:
        raise ValueError('An invalid data resolution was entered - ', data_resolution)

def ci_regions_from_data_resolution(data_resolution):

    if data_resolution == 'patch':
        return [(1, 7, 3, 30), (17, 23, 3, 30)]
    elif data_resolution == 'lowres':
        return 0.1
    elif data_resolution == 'midres':
        return 0.05
    elif data_resolution == 'highres':
        return 0.03
    else:
        raise ValueError('An invalid data resolution was entered - ', data_resolution)

def frame_geometry_from_data_resolution(data_resolution):

    if data_resolution == 'patch':
        return CIFrameIntegration.patch()
    elif data_resolution == 'lowres':
        return CIFrameIntegration.low_res()
    elif data_resolution == 'midres':
        return CIFrameIntegration.mid_res()
    elif data_resolution == 'highres':
        return CIFrameIntegration.high_res()
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_resolution)

def data_resolution_from_shape(shape):

    if shape == (36, 36):
        return 'patch'
    elif shape == (120, 120):
        return 'lowres'
    elif shape == (300, 300):
        return 'midres'
    elif shape == (600, 600):
        return 'highres'
    else:
        raise ValueError('An invalid shape was entered when generating the data-type - ', shape)

def load_test_ci_data(data_name, data_resolution, normalization):

    frame_geometry = frame_geometry_from_data_resolution(data_resolution=data_resolution)
    ci_regions = ci_regions_from_data_resolution(data_resolution=data_resolution)

    pattern = ci_pattern.CIPatternUniform(normalization=normalization, regions=ci_regions)

    data_path = path + '/data/' + data_name + '/' + data_resolution

    normalization = str(int(pattern.normalization))

    return ci_data.ci_data_from_fits(frame_geometry=frame_geometry, ci_pattern=pattern,
                                     image_path=data_path + '/image_' + normalization + '.fits',
                                     noise_map_path=data_path + '/noise_map_' + normalization + '.fits',
                                     ci_pre_cti_path=data_path + '/ci_pre_cti_' + normalization + '.fits')

class CIFrameIntegration(ci_frame.FrameGeometry):

    def __init__(self, corner, parallel_overscan, serial_prescan, serial_overscan):
        """This class represents the quadrant geometry of an integration quadrant."""
        super(CIFrameIntegration, self).__init__(corner=corner, parallel_overscan=parallel_overscan,
                                                 serial_prescan=serial_prescan, serial_overscan=serial_overscan)

    @classmethod
    def patch(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                  serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                  serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def low_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                  serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                  serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def mid_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                  serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                  serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def high_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        return CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                  serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                  serial_prescan=ci_frame.Region((0, 36, 0, 1)))