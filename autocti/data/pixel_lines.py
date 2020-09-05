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
            self.lines = np.array(lines)

            # Extract the attributes from each line
            self.data = np.array([line.data for line in self.lines])
            self.origins = np.array([line.origin for line in self.lines])
            self.locations = np.array([line.location for line in self.lines])
            self.dates = np.array([line.date for line in self.lines])
            self.backgrounds = np.array([line.background for line in self.lines])
            self.fluxes = np.array([line.flux for line in self.lines])
        # Creat the line objects from the inputs
        else:
            self.data = np.array(data)

            n_lines = len(self.data)

            # Default None if not provided
            if origins is None:
                self.origins = np.array([None] * n_lines)
            else:
                self.origins = np.array(origins)
            if locations is None:
                self.locations = np.array([None] * n_lines)
            else:
                self.locations = np.array(locations)
            if dates is None:
                self.dates = np.array([None] * n_lines)
            else:
                self.dates = np.array(dates)
            if backgrounds is None:
                self.backgrounds = np.array([None] * n_lines)
            else:
                self.backgrounds = np.array(backgrounds)
            if fluxes is None:
                self.fluxes = np.array([None] * n_lines)
            else:
                self.fluxes = np.array(fluxes)

            self.lines = np.array(
                [
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
            )

        self.lengths = np.array([line.length for line in self.lines])

    @property
    def n_lines(self):
        return len(self.lines)

    def find_consistent_lines(self, fraction_present=2 / 3):
        """ Identify lines that are consistently present across several images.
        
        This helps to identify warm pixels by discarding noise peaks.
        
        Parameters
        ----------
        self : PixelLineCollection
            Must contain lines from multiple images as identified by their 
            PixelLine.origin with potentially matching lines with the same 
            PixelLine.location in their images.
        
        fraction_present : float 
            The minimum fraction of images in which the pixel must be present.
            
        Returns
        -------
        consistent_lines : [int]
            The indices of consistently present pixel lines in the attribute 
            arrays.
        """
        # The possible locations of warm pixels
        unique_locations = np.unique(self.locations, axis=0)

        # Number of separate images
        n_images = len(np.unique(self.origins))

        # Find consistent lines
        consistent_lines = []
        for loc in unique_locations:
            # Indices of lines with locations matching both the row and column
            found_indices = np.argwhere(
                np.sum(self.locations == loc, axis=1) == 2
            ).flatten()

            # Record line indices if enough warm pixels match that location
            if len(found_indices) / n_images >= fraction_present:
                consistent_lines = np.concatenate((consistent_lines, found_indices))

        return np.unique(consistent_lines).astype(int)
