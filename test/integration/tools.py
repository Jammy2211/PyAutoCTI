import shutil
from os import path

from autofit import conf

from autocti.data import cti_image
from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_frame, ci_pattern
from autocti.tools import infoio

directory = path.dirname(path.realpath(__file__))

dirpath = "{}".format(directory)


class QuadGeometryIntegration(cti_image.FrameGeometry):

    def __init__(self):
        """This class represents the frame_geometry of a Euclid quadrant in the bottom-left of a CCD (see \
        **QuadGeometryEuclid** for a description of the Euclid CCD / FPA)"""

        super(QuadGeometryIntegration, self).__init__(parallel_overscan=cti_image.Region((1, 30, 33, 36)),
                                                      serial_prescan=cti_image.Region((0, 33, 0, 1)),
                                                      serial_overscan=cti_image.Region((0, 33, 30, 36)))

    @staticmethod
    def rotate_before_parallel_cti(image_pre_clocking):
        """ Rotate the quadrant image data before clocking via cti_settings in the parallel direction.

        For the integration quadrant, no rotation is required for parallel clocking

        Params
        ----------
        image_pre_clocking : ndarray
            The ci_pre_ctis before parallel clocking, therefore before it has been reoriented for clocking.
        """
        return image_pre_clocking

    @staticmethod
    def rotate_after_parallel_cti(image_post_clocking):
        """ Re-rotate the quadrant image data after clocking via cti_settings in the parallel direction.

        For the integration quadrant, no re-rotation is required for parallel clocking.

        Params
        ----------
        image_post_clocking : ndarray
            The ci_pre_ctis after clocking, therefore with parallel cti added or corrected.

        """
        return image_post_clocking

    @staticmethod
    def rotate_before_serial_cti(image_pre_clocking):
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

    @staticmethod
    def rotate_after_serial_cti(image_post_clocking):
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

    @staticmethod
    def parallel_trail_from_y(y, dy):
        """Coordinates of a parallel trail of size dy from coordinate y"""
        return y, y + dy + 1

    @staticmethod
    def serial_trail_from_x(x, dx):
        """Coordinates of a serial trail of size dx from coordinate x"""
        return x, x + dx + 1


class CIQuadGeometryIntegration(QuadGeometryIntegration, ci_frame.CIQuadGeometry):

    def __init__(self):
        """This class represents the quadrant geometry of an integration quadrant."""
        super(CIQuadGeometryIntegration, self).__init__()

    @staticmethod
    def parallel_front_edge_region(region, rows=(0, 1)):
        ci_frame.check_parallel_front_edge_size(region, rows)
        return cti_image.Region((region.y0 + rows[0], region.y0 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_trails_region(region, rows=(0, 1)):
        return cti_image.Region((region.y1 + rows[0], region.y1 + rows[1], region.x0, region.x1))

    @staticmethod
    def parallel_side_nearest_read_out_region(region, image_shape, columns=(0, 1)):
        return cti_image.Region((0, image_shape[0], region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_front_edge_region(region, columns=(0, 1)):
        ci_frame.check_serial_front_edge_size(region, columns)
        return cti_image.Region((region.y0, region.y1, region.x0 + columns[0], region.x0 + columns[1]))

    @staticmethod
    def serial_trails_region(region, columns=(0, 1)):
        return cti_image.Region((region.y0, region.y1, region.x1 + columns[0], region.x1 + columns[1]))

    @staticmethod
    def serial_ci_region_and_trails(region, image_shape, from_column):
        return cti_image.Region((region.y0, region.y1, from_column + region.x0, image_shape[1]))


shape = (36, 36)
ci_regions = [(1, 7, 1, 30), (17, 23, 1, 30)]
normalizations = [84700.0]
frame_geometry = CIQuadGeometryIntegration()


def simulate_integration_quadrant(data_name, cti_params, cti_settings):
    data_path = "{}/data/integration/{}".format(dirpath, data_name)
    infoio.make_path_if_does_not_exist(data_path)

    sim_ci_patterns = ci_pattern.create_uniform_simulate_via_lists(normalizations=normalizations, regions=ci_regions)

    sim_ci_datas = list(map(lambda pattern:
                            ci_data.CIImage.simulate(shape=shape, frame_geometry=frame_geometry,
                                                     ci_pattern=pattern, cti_settings=cti_settings,
                                                     cti_params=cti_params,
                                                     read_noise=None),
                            sim_ci_patterns))

    list(map(lambda sim_ci_data, index: sim_ci_data.output_as_fits(path=data_path, filename='/ci_data_' + str(index)),
             sim_ci_datas, range(len(sim_ci_datas))))


def load_ci_datas(data_name):
    data_path = "{}/data/integration/{}".format(dirpath, data_name)

    ci_patterns = ci_pattern.create_uniform_via_lists(normalizations=normalizations, regions=ci_regions)

    images = list(map(lambda pattern, index:
                      ci_data.CIImage.from_fits_and_ci_pattern(path=data_path, filename='/ci_data_' + str(index), hdu=0,
                                                               frame_geometry=frame_geometry, ci_pattern=pattern),
                      ci_patterns, range(len(ci_patterns))))

    noises = list(map(lambda pattern:
                      ci_frame.CIFrame.from_single_value(value=1.0, shape=shape, frame_geometry=frame_geometry,
                                                         ci_pattern=pattern), ci_patterns))

    masks = list(map(lambda pattern:
                     ci_data.CIMask.create(frame_geometry=frame_geometry, ci_pattern=pattern, shape=shape),
                     ci_patterns))

    ci_pre_ctis = list(map(lambda ci_image: ci_image.create_ci_pre_cti(), images))

    return ci_data.CIData(images, masks, noises, ci_pre_ctis)


def reset_paths(data_name, pipeline_name, output_path):
    conf.instance.output_path = output_path

    try:
        shutil.rmtree(dirpath + '/data' + data_name)
    except FileNotFoundError:
        pass

    try:
        shutil.rmtree(output_path + '/' + pipeline_name)
    except FileNotFoundError:
        pass
