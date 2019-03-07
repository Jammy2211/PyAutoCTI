from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern

import os

path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

def shape_from_data_resolution(data_resolution):

    if data_resolution == 'Patch':
        return (36, 36)
    elif data_resolution == 'Low_Res':
        return (120, 120)
    elif data_resolution == 'Mid_Res':
        return (300, 300)
    elif data_resolution == 'High_Res':
        return (600, 600)
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_resolution)

def ci_regions_from_data_resolution(data_resolution):

    if data_resolution == 'Patch':
        return [(1, 7, 3, 30), (17, 23, 3, 30)]
    elif data_resolution == 'Low_Res':
        return 0.1
    elif data_resolution == 'Mid_Res':
        return 0.05
    elif data_resolution == 'High_Res':
        return 0.03
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_resolution)

def frame_geometry_from_data_resolution(data_resolution):

    if data_resolution == 'Patch':
        return CIFrameIntegration.patch()
    elif data_resolution == 'Low_Res':
        return CIFrameIntegration.low_res()
    elif data_resolution == 'Mid_Res':
        return CIFrameIntegration.mid_res()
    elif data_resolution == 'High_Res':
        return CIFrameIntegration.high_res()
    else:
        raise ValueError('An invalid data-type was entered when generating the test-data suite - ', data_resolution)

def data_resolution_from_shape(shape):

    if shape == (36, 36):
        return 'Patch'
    elif shape == (120, 120):
        return 'Low_Res'
    elif shape == (300, 300):
        return 'Mid_Res'
    elif shape == (600, 600):
        return 'High_Res'
    else:
        raise ValueError('An invalid shape was entered when generating the data-type - ', shape)

def load_test_ci_data(data_resolution, data_name):

    pixel_scale = _from_data_type(data_type=data_resolution)

    return ci_data.load_ci_data_from_fits(frame_geometry=frame_geometry, ci_pattern=pattern,
                                          image_path=path + '/data/' + test_name + '/ci_image_0.fits',
                                          noise_map_from_single_value=1.0)


class CIFrameIntegration(ci_frame.FrameGeometry):

    def __init__(self, corner, parallel_overscan, serial_prescan, serial_overscan):
        """This class represents the quadrant geometry of an integration quadrant."""
        super(CIFrameIntegration, self).__init__(corner=corner, parallel_overscan=parallel_overscan,
                                                 serial_prescan=serial_prescan, serial_overscan=serial_overscan)

    @classmethod
    def patch(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                           serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                           serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def low_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                           serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                           serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def mid_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                           serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                           serial_prescan=ci_frame.Region((0, 36, 0, 1)))

    @classmethod
    def high_res(cls):
        """This class represents the quadrant geometry of an integration quadrant."""
        CIFrameIntegration(corner=(0, 0), parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                           serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                           serial_prescan=ci_frame.Region((0, 36, 0, 1)))