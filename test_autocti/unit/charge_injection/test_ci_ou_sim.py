import numpy as np
import pytest
import autocti as ac
from autocti.charge_injection import ci_ou_sim


class TestCIOUSim:
    def test__non_uniform_array_is_correct_with_rotation(self):

        # bottom left

        frame = ci_ou_sim.non_uniform_frame_for_ou_sim(
            ccd_id="123", quadrant_id="E", ci_normalization=50000.0
        )

        assert frame.shape == (2086, 2128)
        assert frame[0, 50] == 0
        assert frame[0, 2099] == 0
        assert (frame[0:450, 51:2099] > 0).all()
        assert 49000.0 < np.mean(frame[0:450, 51:2099]) < 51000.0

        # top left

        frame = ci_ou_sim.non_uniform_frame_for_ou_sim(
            ccd_id="123", quadrant_id="H", ci_normalization=50000.0
        )

        assert frame[1678, 50] == 0
        assert frame[1678, 2099] == 0
        assert (frame[1678:2128, 51:2099] > 0).all()
        assert 49000.0 < np.mean(frame[1678:2128, 51:2099]) < 51000.0

        # bottom right

        frame = ci_ou_sim.non_uniform_frame_for_ou_sim(
            ccd_id="123", quadrant_id="F", ci_normalization=50000.0
        )

        # top right

        assert frame.shape == (2086, 2128)
        assert frame[0, 28] == 0
        assert frame[0, 2077] == 0
        assert (frame[0:450, 29:2077] > 0).all()
        assert 49000.0 < np.mean(frame[0:450, 51:2099]) < 51000.0

        frame = ci_ou_sim.non_uniform_frame_for_ou_sim(
            ccd_id="123", quadrant_id="G", ci_normalization=50000.0
        )

        assert frame.shape == (2086, 2128)
        assert frame[1678, 28] == 0
        assert frame[1678, 2077] == 0
        assert (frame[1678:2128, 29:2077] > 0).all()
        assert 49000.0 < np.mean(frame[1678:2128, 29:2077]) < 51000.0
