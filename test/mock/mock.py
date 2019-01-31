import numpy as np


class MockRegion(tuple):

    def __new__(cls, region):
        region = super(MockRegion, cls).__new__(cls, region)

        region.y0 = region[0]
        region.y1 = region[1]
        region.x0 = region[2]
        region.x1 = region[3]

        return region

    def set_region_on_array_to_zeros(self, array):
        array[self.y0:self.y1, self.x0:self.x1] = 0.0
        return array


class MockPattern(object):

    def __init__(self):
        pass


class MockGeometry(object):

    def __init__(self):
        super(MockGeometry, self).__init__()


class MockCIGeometry(object):

    def __init__(self, serial_prescan=(0, 1, 0, 1), serial_overscan=(0, 1, 0, 1)):
        super(MockCIGeometry, self).__init__()
        self.serial_prescan = MockRegion(serial_prescan)
        self.serial_overscan = MockRegion(serial_overscan)


class MockCIPreCTI(np.ndarray):

    def __new__(cls, array, frame_geometry=MockGeometry(), ci_pattern=MockPattern(), value=1.0, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = frame_geometry
        ci.ci_pattern = ci_pattern
        ci.value = value
        return ci

    def ci_post_cti_from_cti_params_and_settings(self, cti_params, cti_settings):
        return self.value * np.ones((2, 2))


class MockParams(object):

    def __init__(self):
        pass


class MockSettings(object):

    def __init__(self):
        pass


class MockChInj(np.ndarray):

    def __new__(cls, array, geometry=None, ci_pattern=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ci.frame_geometry = geometry
        ci.ci_pattern = ci_pattern
        return ci