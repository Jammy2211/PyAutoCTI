import numpy as np


class PixelLine(object):
    def __init__(
        self,
        data=None,
        origin=None,
        location=None,
        date=None,
        background=None,
        flux=None,
        n_stacked=None,
    ):
        """ A 1D line of pixels (e.g. a single CTI trail) with metadata.
        
        Or could be an averaged stack of many lines, in which case the metadata 
        parameters may be e.g. the average value or the minimum value of a bin.
        
        Parameters
        ----------
        data : [float]
            The pixel counts, in units of electrons.
            
        origin : str
            An identifier for the origin (e.g. image name) of the data.
            
        location : [int, int] 
            The row and column indices of the first pixel in the line in the 
            image. The row index is the distance in pixels to the readout 
            register minus one.
            
        date : float 
            The Julian date.
            
        background : float
            The background charge count, in units of electrons. It is assumed 
            that the background has already been subtracted from the data.
            
        flux : float
            The maximum charge in the line, or e.g. for a CTI trail the original 
            flux before trailing, in units of electrons.
            
        n_stacked : int
            If the line is an averaged stack, the number of stacked lines.
            
        Attributes
        ----------
        length : int 
            The number of pixels in the data array.
        """
        self.data = data
        self.origin = origin
        self.location = location
        self.date = date
        self.background = background
        self.flux = flux
        self.n_stacked = n_stacked

        # Default flux from data
        if self.flux is None and self.data is not None:
            self.flux = np.amax(self.data)

    @property
    def length(self):
        if self.data is not None:
            return len(self.data)
        else:
            return None


class PixelLineCollection(object):
    def __init__(
        self,
        lines=None,
        data=None,
        origins=None,
        locations=None,
        dates=None,
        backgrounds=None,
        fluxes=None,
    ):
        """ A collection of 1D lines of pixels with metadata.
        
        Enables convenient analysis e.g. binning and stacking of CTI trails.
        
        Either provide a list of existing PixelLine objects, or lists of the 
        individual parameters for PixelLine objects.
        
        Parameters
        ----------
        lines : [PixelLine]
            A list of the PixelLine objects.
            
        # or
        
        data : [[float]] 
            The pixel counts of each line, in units of electrons.
            
        origins : [str]
            The identifiers for the origins (e.g. image name) of each line.
            
        locations : [[int, int]] 
            The row and column indices of the first pixel in the line in the 
            image, for each line.
            
        dates : [float] 
            The Julian date of each line.
            
        backgrounds : [float]
            The background charge count of each line, in units of electrons.
            
        fluxes : [float]
            The maximum charge in each line, in units of electrons.
            
        Attributes
        ----------
        lengths : [int] 
            The number of pixels in the data array of each line.
            
        n_lines : int 
            The number of lines in the collection.
        """
        # Extract the attributes from each line object
        if lines is not None:
            self.lines = lines

            # Extract the attributes from each line
            self.data = [line.data for line in self.lines]
            self.origins = [line.origin for line in self.lines]
            self.locations = [line.location for line in self.lines]
            self.dates = [line.date for line in self.lines]
            self.backgrounds = [line.background for line in self.lines]
            self.fluxes = [line.flux for line in self.lines]
        # Creat the line objects from the inputs
        else:
            self.data = data

            n_lines = len(self.data)

            # Default None if not provided
            if origins is None:
                self.origins = [None] * n_lines
            else:
                self.origins = origins
            if locations is None:
                self.locations = [None] * n_lines
            else:
                self.locations = locations
            if dates is None:
                self.dates = [None] * n_lines
            else:
                self.dates = dates
            if backgrounds is None:
                self.backgrounds = [None] * n_lines
            else:
                self.backgrounds = backgrounds
            if fluxes is None:
                self.fluxes = [None] * n_lines
            else:
                self.fluxes = fluxes

            self.lines = [
                PixelLine(
                    data=data,
                    origin=origin,
                    location=location,
                    date=date,
                    background=background,
                    flux=flux,
                )
                for data, origin, location, date, background, flux in zip(
                    self.data,
                    self.origins,
                    self.locations,
                    self.dates,
                    self.backgrounds,
                    self.fluxes,
                )
            ]

        self.lengths = [line.length for line in self.lines]

    @property
    def n_lines(self):
        return len(self.lines)
