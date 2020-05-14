from autofit import exc


class ArrayException(Exception):
    pass


class MaskException(Exception):
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
