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
    def __init__(self, lines):
        """ A collection of 1D lines of pixels with metadata.
        
        Enables convenient analysis e.g. binning and stacking of CTI trails.
        
        Parameters
        ----------
        lines : [PixelLine]
            A list of the PixelLine objects.
            
        Attributes
        ----------
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
        
        lengths : [int] 
            The number of pixels in the data array of each line.
            
        n_lines : int 
            The number of lines in the collection.
        """
        # Extract the attributes from each line object
        self.lines = lines

        # Extract the attributes from each line
        self.data = [line.data for line in lines]
        self.origins = [line.origin for line in lines]
        self.locations = [line.location for line in lines]
        self.dates = [line.date for line in lines]
        self.backgrounds = [line.background for line in lines]
        self.fluxes = [line.flux for line in lines]

    @property
    def n_lines(self):
        return len(self.lines)
