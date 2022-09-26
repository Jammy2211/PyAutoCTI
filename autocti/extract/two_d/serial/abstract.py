from autocti.extract.two_d.abstract import Extract2D


class Extract2DSerial(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn 2D serial data (E.g. an EPER) into 1D data.

        For a serial extract `axis=0` such that binning is performed over the columns containing the EPER.
        """
        return 0
