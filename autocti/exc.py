from autofit import exc
from autoarray.exc import ArrayException, MaskException, FrameException, RegionException


class DataException(Exception):
    pass


class CIPatternException(Exception):
    pass


class PlottingException(Exception):
    pass


class FittingException(Exception):
    pass


class PriorException(exc.FitException):
    pass
