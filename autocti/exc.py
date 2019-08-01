from autofit import exc


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
