import numpy as np


class MockPattern(object):
    def __init__(self, regions=None):

        self.normalization = 10
        self.regions = regions
        self.total_rows = 2
        self.total_columns = 2


class MockGeometry(object):
    def __init__(self):
        super(MockGeometry, self).__init__()


class MockFrameGeometry(object):
    def __init__(self, value=1.0):
        self.value = value

    def add_cti(self, image, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockCIFrame(object):
    def __init__(self, value=1.0):

        self.ci_pattern = MockPattern()
        self.frame_geometry = MockFrameGeometry(value=value)
        self.value = value

    def ci_regions_from_array(self, array):
        return array[0:2, 0]

    def parallel_non_ci_regions_frame_from_frame(self, array):
        return array[0:2, 1]

    def serial_all_trails_frame_from_frame(self, array):
        return array[0, 0:2]

    def serial_overscan_no_trails_frame_from_frame(self, array):
        return array[1, 0:2]

    def parallel_front_edge_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def parallel_trails_line_binned_over_columns_from_frame(
        self, array, rows=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def serial_front_edge_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])

    def serial_trails_line_binned_over_rows_from_frame(
        self, array, columns=None, mask=None
    ):
        return np.array([1.0, 1.0, 2.0, 2.0])


class MockCIPreCTI(np.ndarray):
    def __new__(
        cls,
        array,
        frame_geometry=MockGeometry(),
        ci_pattern=MockPattern(),
        value=1.0,
        *args,
        **kwargs
    ):
        ci = np.array(array).view(cls)
        ac.ci.frame_geometry = frame_geometry
        ac.ci.ci_pattern = ci_pattern
        ac.ci.value = value
        return ci

    def ci_post_cti_from_cti_params_and_settings(self, cti_params, clocker):
        return self.value * np.ones((2, 2))


class MockChInj(np.ndarray):
    def __new__(cls, array, geometry=None, ci_pattern=None, *args, **kwargs):
        ci = np.array(array).view(cls)
        ac.ci.frame_geometry = geometry
        ac.ci.ci_pattern = ci_pattern
        return ci
