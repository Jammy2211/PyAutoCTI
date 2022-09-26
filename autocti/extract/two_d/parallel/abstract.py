from autocti.extract.two_d.abstract import Extract2D


class Extract2DParallel(Extract2D):
    @property
    def binning_axis(self) -> int:
        """
        The axis over which binning is performed to turn 2D parallel data (e.g. an EPER) into 1D data.

        For a parallel extract `axis=1` such that binning is performed over the rows containing the EPER.
        """
        return 1
