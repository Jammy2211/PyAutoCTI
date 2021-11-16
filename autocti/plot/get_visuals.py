import autoarray.plot as aplt

from autoarray.plot.mat_wrap import get_visuals as gv

from autocti.charge_injection.fit import FitImagingCI


class GetVisuals1D(gv.GetVisuals1D):
    def __init__(self, include: aplt.Include1D, visuals: aplt.Visuals1D):
        """
        Class which gets 1D attributes and adds them to a `Visuals1D` objects, such that they are plotted on 1D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include1D` object. If this entry is `False`, the `GetVisuals1D.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals1D` class adds new visuals to a pre-existing `Visuals1D` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 1D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals1D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals1D` class.
        """
        super().__init__(include=include, visuals=visuals)


class GetVisuals2D(gv.GetVisuals2D):
    def __init__(self, include: aplt.Include2D, visuals: aplt.Visuals2D):
        """
        Class which gets 2D attributes and adds them to a `Visuals2D` objects, such that they are plotted on 2D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include2D` object. If this entry is `False`, the `GetVisuals2D.get` method returns a None and the
        attribute is omitted from the plot.

        The `GetVisuals2D` class adds new visuals to a pre-existing `Visuals2D` object that is passed to
        its `__init__` method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 2D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals2D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals2D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def via_fit_imaging_ci_from(self, fit: FitImagingCI) -> aplt.Visuals2D:
        """
        From a `FitImagingCI` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `FitImagingCI` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 2D coordinate system.
        - mask: the 2D mask.
        - border: the border of the 2D mask, which are all of the mask's exterior edge pixels.
        - parallel overscan: the 2D region defining the parallel overscan on the imaging data.
        - serial prescan: the 2D region defining the serial prerscan on the imaging data.
        - serial overscan: the 2D region defining the serial overscan on the imaging data.

        Parameters
        ----------
        fit
            The fit imaging object whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        visuals_2d_via_fit = super().via_fit_imaging_from(fit=fit)

        parallel_overscan = self.get(
            "parallel_overscan", fit.imaging_ci.layout.parallel_overscan
        ),
        serial_prescan = self.get(
            "serial_prescan", fit.imaging_ci.layout.serial_prescan
        ),
        serial_overscan = self.get(
            "serial_overscan", fit.imaging_ci.layout.serial_overscan
        ),

        return visuals_2d_via_fit + self.visuals.__class__(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan
        )
