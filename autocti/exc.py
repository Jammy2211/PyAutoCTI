from autofit import exc
from autoarray.exc import ArrayException, MaskException, RegionException


class DatasetException(Exception):
    pass


class PatternLineException(Exception):
    pass


class ImagingCIException(Exception):
    pass


class ExtractException(Exception):
    pass


class LayoutException(Exception):
    pass


class PlottingException(Exception):
    pass


class FittingException(Exception):
    pass


class PriorException(exc.FitException):
    pass


class ClockerException(Exception):
    pass
