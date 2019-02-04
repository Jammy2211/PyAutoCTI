import os
import shutil

from autocti.charge_injection import ci_data, ci_frame, ci_pattern
from autocti.data import util

dirpath = os.path.dirname(os.path.realpath(__file__))


def reset_paths(test_name, output_path):
    try:
        shutil.rmtree(dirpath + '/data/' + test_name)
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree(output_path + '/' + test_name)
    except FileNotFoundError:
        pass


class CIQuadGeometryIntegration(ci_frame.FrameGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of an integration quadrant."""
        super(CIQuadGeometryIntegration, self).__init__(parallel_overscan=ci_frame.Region((33, 36, 1, 30)),
                                                        serial_overscan=ci_frame.Region((0, 33, 31, 36)),
                                                        serial_prescan=ci_frame.Region((0, 36, 0, 1)), corner=(0, 0))

    def rotate_for_parallel_cti(self, image):
        """ Rotate the quadrant image data before clocking via cti_settings in the parallel direction.

        For the integration quadrant, no rotation is required for parallel clocking

        Params
        ----------
        image_pre_clocking : ndarray
            The ci_pre_ctis before parallel clocking, therefore before it has been reoriented for clocking.
        """
        return image

    def rotate_before_serial_cti(self, image_pre_clocking):
        """ Rotate the quadrant image data before clocking via cti_settings in the serial direction.

        For the integration quadrant, the image is rotated 90 degrees anti-clockwise for serial clocking.

        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable data-type for C++ (and therefore cti_settings), so we use .copy() to force a memory re-ordering.

        Params
        ----------
        image_pre_clocking : ndarray
            The ci_pre_ctis before serial clocking, therefore before it has been reoriented for clocking.
        """

        return image_pre_clocking.T.copy()

    def rotate_after_serial_cti(self, image_post_clocking):
        """ Re-rotate the quadrant image data after clocking via cti_settings in the serial direction.

        For the integration quadrant, the ci_pre_ctis is re-rotated 90 degrees clockwise after serial clocking.


        NOTE : The NumPy transpose routine does not reorder the array's memory, making it non-contiguous. This is not \
        a useable data-type for C++ (and therefore cti_settings), so we use .copy() to force a memory re-ordering.

        Params
        ----------
        image_post_clocking : ndarray
            The ci_pre_ctis after clocking, therefore with serial cti added or corrected.

        """
        return image_post_clocking.T.copy()

    def parallel_trail_from_y(self, y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y, y + dy + 1

    def serial_trail_from_x(self, x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x, x + dx + 1


shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
frame_geometry = CIQuadGeometryIntegration()


def simulate_integration_quadrant(test_name, normalizations, cti_params, cti_settings):
    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + test_name + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sim_ci_patterns = ci_pattern.uniform_simulate_from_lists(normalizations=normalizations, regions=ci_regions)

    sim_ci_datas = list(map(lambda pattern:
                            ci_data.simulate(shape=shape, frame_geometry=frame_geometry,
                                             ci_pattern=pattern, cti_settings=cti_settings,
                                             cti_params=cti_params,
                                             read_noise=None),
                            sim_ci_patterns))

    list(map(lambda sim_ci_data, index:
             util.numpy_array_to_fits(array=sim_ci_data, file_path=output_path + '/ci_data_' + str(index) + '.fits'),
             sim_ci_datas, range(len(sim_ci_datas))))
