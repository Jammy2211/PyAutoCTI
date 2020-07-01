from autofit import exc
from autoarray.exc import MaskException

class ArrayException(Exception):
    pass


class DataException(Exception):
    pass


class CIPatternException(Exception):
    pass


class RegionException(Exception):
    pass


class PlottingException(Exception):
    pass


class FittingException(Exception):
    pass


class PriorException(exc.FitException):
    pass


class FrameException(Exception):
    pass
