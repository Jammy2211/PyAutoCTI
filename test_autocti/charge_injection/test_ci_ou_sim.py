import numpy as np
from autocti.charge_injection import ci_ou_sim


class TestCIOUSim:
    def test__non_uniform_array_is_correct_with_rotation(self):

        # bottom left

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="E", ci_normalization=50000.0
        )

        assert frame.shape == (2086, 2128)
        assert frame[0, 50] == 0
        assert frame[0, 2099] == 0
        assert (frame[0:200, 51:2099] > 0).all()
        assert 49000.0 < np.mean(frame[0:200, 51:2099]) < 51000.0

        # top left

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="H", ci_normalization=50000.0
        )

        assert frame[1938, 50] == 0
        assert frame[1938, 2099] == 0
        assert (frame[1928:2128, 51:2099] > 0).all()
        assert 49000.0 < np.mean(frame[1928:2128, 51:2099]) < 51000.0

        # bottom right

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="F", ci_normalization=50000.0
        )

        assert frame.shape == (2086, 2128)
        assert frame[0, 28] == 0
        assert frame[0, 2077] == 0
        assert (frame[0:200, 29:2077] > 0).all()
        assert 49000.0 < np.mean(frame[0:200, 51:2099]) < 51000.0

        # top right

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="G", ci_normalization=50000.0
        )

        assert frame.shape == (2086, 2128)
        assert frame[1938, 28] == 0
        assert frame[1938, 2077] == 0
        assert (frame[1928:2128, 29:2077] > 0).all()
        assert 49000.0 < np.mean(frame[1928:2128, 29:2077]) < 51000.0

    def test__add_cti_to_ci_pre_cti(self):

        # bottom left

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="E", ci_normalization=50000.0
        )

        assert frame[199, 100] > 0.0
        assert frame[200, 100] == 0.0

        post_cti_ci = ci_ou_sim.add_cti_to_ci_pre_cti(
            pre_cti_ci=frame[:, 100:101], ccd_id="123", quadrant_id="E"
        )

        assert post_cti_ci[200, 0] > 0.0

        # top left

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="H", ci_normalization=50000.0
        )

        assert frame[1886, 100] > 0.0
        assert frame[1885, 100] == 0.0

        post_cti_ci = ci_ou_sim.add_cti_to_ci_pre_cti(
            pre_cti_ci=frame[:, 100:101], ccd_id="123", quadrant_id="H"
        )

        assert post_cti_ci[1885, 0] > 0.0

        # bottom right

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="F", ci_normalization=50000.0
        )

        assert frame[199, 100] > 0.0
        assert frame[200, 100] == 0.0

        post_cti_ci = ci_ou_sim.add_cti_to_ci_pre_cti(
            pre_cti_ci=frame[:, 100:101], ccd_id="123", quadrant_id="F"
        )

        assert post_cti_ci[200, 0] > 0.0

        # top right

        frame = ci_ou_sim.non_uniform_frame_from(
            ccd_id="123", quadrant_id="G", ci_normalization=50000.0
        )

        assert frame[1886, 100] > 0.0
        assert frame[1885, 100] == 0.0

        post_cti_ci = ci_ou_sim.add_cti_to_ci_pre_cti(
            pre_cti_ci=frame[:, 100:101], ccd_id="123", quadrant_id="G"
        )

        assert post_cti_ci[1885, 0] > 0.0
