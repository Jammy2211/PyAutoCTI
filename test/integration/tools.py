import os
import shutil
from os import path

from autofit import conf

from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_frame
from autocti.data.charge_injection import ci_pattern

import os

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

    # def parallel_front_edge_region(self, region, rows=(0, 1)):
    #     ci_frame.check_parallel_front_edge_size(region, rows)
    #     return ci_frame.Region((region.y0 + rows[0], region.y0 + rows[1], region.x0, region.x1))
    #
    # def parallel_trails_region(self, region, rows=(0, 1)):
    #     return ci_frame.Region((region.y1 + rows[0], region.y1 + rows[1], region.x0, region.x1))
    #
    # def parallel_side_nearest_read_out_region(self, region, image_shape, columns=(0, 1)):
    #     return ci_frame.Region((0, image_shape[0], region.x0 + columns[0], region.x0 + columns[1]))
    #
    # def serial_front_edge_region(self, region, columns=(0, 1)):
    #     ci_frame.check_serial_front_edge_size(region, columns)
    #     return ci_frame.Region((region.y0, region.y1, region.x0 + columns[0], region.x0 + columns[1]))
    #
    # def serial_trails_region(self, region, columns=(0, 1)):
    #     return ci_frame.Region((region.y0, region.y1, region.x1 + columns[0], region.x1 + columns[1]))
    #
    # def serial_ci_region_and_trails(self, region, image_shape, column):
    #     return ci_frame.Region((region.y0, region.y1, column + region.x0, image_shape[1]))


shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
frame_geometry = CIQuadGeometryIntegration()


def simulate_integration_quadrant(test_name, cti_params, cti_settings):

    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + test_name + '/'

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    sim_ci_patterns = ci_pattern.create_uniform_simulate_via_lists(normalizations=normalizations, regions=ci_regions)

    sim_ci_datas = list(map(lambda pattern:
                            ci_data.CIImage.simulate(shape=shape, frame_geometry=frame_geometry,
                                                     ci_pattern=pattern, cti_settings=cti_settings,
                                                     cti_params=cti_params,
                                                     read_noise=None),
                            sim_ci_patterns))

    list(map(lambda sim_ci_data, index: sim_ci_data.output_as_fits(path=output_path, filename='/ci_data_' + str(index)),
             sim_ci_datas, range(len(sim_ci_datas))))