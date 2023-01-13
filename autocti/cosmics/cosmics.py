import astropy.io.fits as pyfits
import numpy as np
from os import path
from scipy.interpolate import interp1d
from typing import Tuple

import autoarray as aa


class SimulatorCosmicRayMap:
    def __init__(
        self,
        shape_native: Tuple[int, int],
        lengths: np.ndarray,
        distances: np.ndarray,
        flux_scaling: float = 1.0,
        pixel_scale: float = 0.1,
        seed=-1,
    ):
        """
        Returns a map of cosmic rays.

        Adapted from the OU-SIM ELVIS Simulator in Euclid, originally written by Patrick Hudelot.

        Parameters
        ----------
        shape_native
            The 2D shape of the cosmic ray map, corresponding to the image the cosmic arrays are to be added to.
        flux_scaling
            A factor which scales the overall normalization of the cosmic rays.
        settings_dict
            A dictionary of all settings that control the behaviour of the cosmic ray simulator.
        seed
            Random number seed, set to positive value for reproduceable cosmic ray maps.
        """

        self.shape_native = shape_native

        self.lengths = lengths
        self.distances = distances

        self.settings_dict = {}
        self.flux_scaling = flux_scaling

        self.pixel_scale = pixel_scale

        if seed == -1:
            seed = np.random.randint(
                0, 1e9
            )  # Use one seed, so all regions have identical column non-uniformity.
        np.random.seed(seed)

    @classmethod
    def from_fits(
        cls,
        length_file: str,
        distance_file: str,
        shape_native: Tuple[int, int],
        flux_scaling: float = 1.0,
        pixel_scale: float = 0.1,
        seed=-1,
    ) -> "SimulatorCosmicRayMap":
        """
        Creates a cosmic ray simulator where the length and distance arrays are loaded from user supplied .fits files.

        Parameters
        ----------
        length_file
            A .fits table describing the lengths properties of the cosmic rays that are simulated.
        distance_file
            A .fits table describing the distance properties of the cosmic rays that are simulated.
        shape_native
            The 2D shape of the cosmic ray map, corresponding to the image the cosmic arrays are to be added to.
        flux_scaling
            A factor which scales the overall normalization of the cosmic rays.
        """

        lengths = pyfits.getdata(length_file, extname="DATA")
        lengths = np.array(lengths.tolist())

        distances = pyfits.getdata(distance_file, extname="DATA")
        distances = np.array(distances.tolist())

        return SimulatorCosmicRayMap(
            shape_native=shape_native,
            lengths=lengths,
            distances=distances,
            flux_scaling=flux_scaling,
            pixel_scale=pixel_scale,
            seed=seed,
        )

    @classmethod
    def defaults(
        cls,
        shape_native: Tuple[int, int],
        flux_scaling: float = 1.0,
        pixel_scale: float = 0.1,
        seed=-1,
    ) -> "SimulatorCosmicRayMap":
        """
        Creates a cosmic ray simulator where the length and distance arrays are loaded from the default files stored
        on the `PyAutoCTI` repository in `autocti.cosmics`.

        Parameters
        ----------
        length_file
            A .fits table describing the lengths properties of the cosmic rays that are simulated.
        distance_file
            A .fits table describing the distance properties of the cosmic rays that are simulated.
        shape_native
            The 2D shape of the cosmic ray map, corresponding to the image the cosmic arrays are to be added to.
        flux_scaling
            A factor which scales the overall normalization of the cosmic rays.
        """

        dir_path = path.dirname(path.realpath(__file__))

        from autocti.cosmics.cr_lengths import lengths
        from autocti.cosmics.cr_distances import distances

        return SimulatorCosmicRayMap(
            shape_native=shape_native,
            lengths=lengths,
            distances=distances,
            flux_scaling=flux_scaling,
            pixel_scale=pixel_scale,
            seed=seed,
        )

    def intercepts_from(self, luminosities, x0, y0, lengths, angles, flux_scaling):
        """
        Derive cosmic ray streak intercept points.

        Parameters
        ----------
        luminosities
            The luminosities of the cosmic ray tracks.
        x0
            Central positions of the cosmic ray tracks in x-direction.
        y0
            Central positions of the cosmic ray tracks in y-direction.
        lengths
            The lengths of the cosmic ray tracks.
        angles
            The orientation angles of the cosmic ray tracks.
        """
        # create empty arrays
        image = np.zeros((self.shape_native[0], self.shape_native[1]), dtype=np.float64)

        # x and y shifts
        dx = lengths * np.cos(angles) / 2.0  # beware! 0<phi< pi, dx < 0
        dy = lengths * np.sin(angles) / 2.0
        mskdx = np.abs(dx) < 1e-8
        mskdy = np.abs(dy) < 1e-8
        dx[mskdx] = 0.0
        dy[mskdy] = 0.0

        # pixels in x-direction
        ilo = np.round(x0.copy() - dx)
        mask = ilo < 0.0
        ilo[mask] = 0
        ilo = ilo.astype(np.int)

        ihi = np.round(x0.copy() + dx)
        mask = ihi > self.shape_native[1]
        ihi[mask] = self.shape_native[1]
        ihi = ihi.astype(np.int)

        # pixels in y-directions
        jlo = np.round(y0.copy() - dy)
        mask = jlo < 0.0
        jlo[mask] = 0
        jlo = jlo.astype(np.int)

        jhi = np.round(y0.copy() + dy)
        mask = jhi > self.shape_native[0]
        jhi[mask] = self.shape_native[0]
        jhi = jhi.astype(np.int)

        offending_delta = 1.0

        # loop over the individual events
        for i, luminosity in enumerate(luminosities):

            n = 0  # count the intercepts

            u = []
            x = []
            y = []

            # Compute X intercepts on the pixel grid
            if ilo[i] < ihi[i]:
                for xcoord in range(ilo[i], ihi[i]):
                    ok = (xcoord - x0[i]) / dx[i]
                    if np.abs(ok) <= offending_delta:
                        n += 1
                        u.append(ok)
                        x.append(xcoord)
                        y.append(y0[i] + ok * dy[i])
            else:
                for xcoord in range(ihi[i], ilo[i]):
                    ok = (xcoord - x0[i]) / dx[i]
                    if np.abs(ok) <= offending_delta:
                        n += 1
                        u.append(ok)
                        x.append(xcoord)
                        y.append(y0[i] + ok * dy[i])

            # Compute Y intercepts on the pixel grid
            if jlo[i] < jhi[i]:
                for ycoord in range(jlo[i], jhi[i]):
                    ok = (ycoord - y0[i]) / dy[i]
                    if np.abs(ok) <= offending_delta:
                        n += 1
                        u.append(ok)
                        x.append(x0[i] + ok * dx[i])
                        y.append(ycoord)
            else:
                for ycoord in range(jhi[i], jlo[i]):
                    ok = (ycoord - y0[i]) / dy[i]
                    if np.abs(ok) <= offending_delta:
                        n += 1
                        u.append(ok)
                        x.append(x0[i] + ok * dx[i])
                        y.append(ycoord)

            # check if no intercepts were found
            if n < 1:
                xc = int(np.floor(x0[i]))
                yc = int(np.floor(y0[i]))
                image[yc, xc] += luminosity * flux_scaling

            # Find the arguments that sort the intersections along the track
            u = np.asarray(u)
            x = np.asarray(x)
            y = np.asarray(y)

            args = np.argsort(u)

            u = u[args]
            x = x[args]
            y = y[args]

            # Decide which cell each interval traverses, and the path length
            for j in range(1, n - 1):
                w = (u[j + 1] - u[j]) / 2.0
                cx = int(1 + np.floor((x[j + 1] + x[j]) / 2.0))
                cy = int(1 + np.floor((y[j + 1] + y[j]) / 2.0))

                if 0 <= cx < self.shape_native[1] and 0 <= cy < self.shape_native[0]:
                    image[cy, cx] += w * luminosity * flux_scaling

        return image

    def cosmic_ray_map_from(
        self, cover_fraction: float = 1.4, limit: float = 1000.0
    ) -> aa.Array2D:
        """
        Return a cosmic ray, where cosmic rays are generated using the lengths and distance of the class instance.


        Parameters
        ----------
        limit
            The limiting energy for the cosmic ray event.
        cover_fraction
            The covering fraction of cosmic rays over the total number of pixels (in percent) normalized for a 5
            65s exposure time.
        """

        # Prepare the CR map
        cosmic_ray_map = np.zeros((self.shape_native[0], self.shape_native[1]))

        # how many events to draw at once, too large number leads to exceeding the covering fraction

        cdf = self.lengths[:, 1]
        ucr = self.lengths[:, 0]
        approx_pdf = (cdf[1:] - cdf[0:-1]) / (ucr[1:] - ucr[0:-1])
        average_length = (approx_pdf * ucr[1:]).sum() / approx_pdf.sum()
        total_guess = (
            cover_fraction
            / 100.0
            * self.shape_native[1]
            * self.shape_native[0]
            / average_length
        )

        # allocating for a max. 5% error in cover. fraction, aprox.
        # Notice that the minimum number of events will be one...
        cr_n = max(int(total_guess * 0.05), 1)

        covering = 0.0
        lengths = []
        energies = []
        total_cosmics = 0

        while covering < cover_fraction:

            # pseudo-random numbers taken from a uniform distribution between 0 and 1
            luck = np.random.rand(cr_n)

            # draw the length of the tracks
            ius = interp1d(self.lengths[:, 1], self.lengths[:, 0], kind="slinear")
            length = ius(luck)

            if limit is None:
                ius = interp1d(
                    self.distances[:, 1], self.distances[:, 0], kind="slinear"
                )
                energy = ius(luck)
            else:
                # set the energy directly to the limit
                energy = np.asarray([limit])

            lengths += length.tolist()
            energies += energy.tolist()

            # Choose the properties such as positions and an angle from a random Uniform dist
            x = self.shape_native[1] * np.random.rand(int(np.floor(cr_n)))
            y = self.shape_native[0] * np.random.rand(int(np.floor(cr_n)))
            angle = np.pi * np.random.rand(int(np.floor(cr_n)))

            # find the intercepts
            cosmic_ray_map += self.intercepts_from(
                energy, x, y, length, angle, self.flux_scaling
            )

            # count the covering factor
            area = np.count_nonzero(cosmic_ray_map)
            covering = 100.0 * area / (self.shape_native[1] * self.shape_native[0])

            total_cosmics += cr_n

        return aa.Array2D.no_mask(
            values=cosmic_ray_map, pixel_scales=self.pixel_scale
        )
