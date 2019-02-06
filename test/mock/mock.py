import numpy as np
from autocti.charge_injection.ci_frame import Region


class MockPattern(object):

    def __init__(self):
        pass


class MockGeometry(object):

    def __init__(self):
        super(MockGeometry, self).__init__()


class MockCIFrame(object):

    def __init__(self, value=1.0):

        self.value = value

    def add_cti(self, image, cti_params, cti_settings):
        return self.value * np.ones((2, 2))

class MockCIGeometry(object):

    def __init__(self, serial_prescan=(0, 1, 0, 1), serial_overscan=(0, 1, 0, 1)):
        super(MockCIGeometry, self).__init__()
        self.serial_prescan = Region(serial_prescan)
        self.serial_overscan = Region(serial_overscan)


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