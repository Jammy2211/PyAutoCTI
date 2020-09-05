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
        
        Or e.g. a stacked average of many lines, in which case the metadata 
        parameters may be e.g. the average value or minimum value of a bin.
        
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
