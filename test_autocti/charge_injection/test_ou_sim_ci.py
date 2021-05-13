import numpy as np
import autocti as ac
from autocti.charge_injection import ou_sim_ci


class TestCIOUSim:
    def test__non_uniform_array_is_correct_with_rotation(self):

        # bottom left

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="E", ci_normalization=50000.0
        )

        assert array.shape == (2086, 2128)
        assert array[0, 50] == 0
        assert array[0, 2099] == 0
        assert (array[0:200, 51:2099] > 0).all()
        assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

        # top left

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="H", ci_normalization=50000.0
        )

        assert array[1938, 50] == 0
        assert array[1938, 2099] == 0
        assert (array[1928:2128, 51:2099] > 0).all()
        assert 49000.0 < np.mean(array[1928:2128, 51:2099]) < 51000.0

        # bottom right

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="F", ci_normalization=50000.0
        )

        assert array.shape == (2086, 2128)
        assert array[0, 28] == 0
        assert array[0, 2077] == 0
        assert (array[0:200, 29:2077] > 0).all()
        assert 49000.0 < np.mean(array[0:200, 51:2099]) < 51000.0

        # top right

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="G", ci_normalization=50000.0
        )

        assert array.shape == (2086, 2128)
        assert array[1938, 28] == 0
        assert array[1938, 2077] == 0
        assert (array[1928:2128, 29:2077] > 0).all()
        assert 49000.0 < np.mean(array[1928:2128, 29:2077]) < 51000.0

    def test__add_cti_to_pre_cti_image(self):

        # bottom left

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="E", ci_normalization=50000.0
        )

        assert array[199, 100] > 0.0
        assert array[200, 100] == 0.0

        pre_cti_image = ac.Array2D.manual_native(array[:, 100:101], pixel_scales=0.1)

        post_cti_image = ou_sim_ci.add_cti_to_pre_cti_image(
            pre_cti_image=pre_cti_image, ccd_id="123", quadrant_id="E"
        )

        assert post_cti_image[200, 0] > 0.0

        # top left

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="H", ci_normalization=50000.0
        )

        assert array[1886, 100] > 0.0
        assert array[1885, 100] == 0.0

        pre_cti_image = ac.Array2D.manual_native(array[:, 100:101], pixel_scales=0.1)

        post_cti_image = ou_sim_ci.add_cti_to_pre_cti_image(
            pre_cti_image=pre_cti_image, ccd_id="123", quadrant_id="H"
        )

        assert post_cti_image[1885, 0] > 0.0

        # bottom right

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="F", ci_normalization=50000.0
        )

        assert array[199, 100] > 0.0
        assert array[200, 100] == 0.0

        pre_cti_image = ac.Array2D.manual_native(array[:, 100:101], pixel_scales=0.1)

        post_cti_image = ou_sim_ci.add_cti_to_pre_cti_image(
            pre_cti_image=pre_cti_image, ccd_id="123", quadrant_id="F"
        )

        assert post_cti_image[200, 0] > 0.0

        # top right

        array = ou_sim_ci.non_uniform_array_from(
            ccd_id="123", quadrant_id="G", ci_normalization=50000.0
        )

        assert array[1886, 100] > 0.0
        assert array[1885, 100] == 0.0

        pre_cti_image = ac.Array2D.manual_native(array[:, 100:101], pixel_scales=0.1)

        post_cti_image = ou_sim_ci.add_cti_to_pre_cti_image(
            pre_cti_image=pre_cti_image, ccd_id="123", quadrant_id="G"
        )

        assert post_cti_image[1885, 0] > 0.0
