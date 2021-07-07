"""
@file python/ELViS/CR.py
@date 07/13/16
@author user
"""

__author__ = "hudelot"

import sys
import os

import astropy.io.fits as pyfits
import numpy as np
import logging
from autocti.cosmics import fitslib

# import scipy
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


class CosmicRays:
    """ Handle cosmic rays simulator """

    def __init__(self, shape, cr_fluxscaling, seed=-1):
        self.image = None

        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.cr_length_file = os.path.join(dir_path, "crlength_v2.fits")
        self.cr_distance_file = os.path.join(dir_path, "crdist.fits")
        self.cr = None
        self.nx = shape[1]
        self.ny = shape[0]
        self.cr_fluxscaling = cr_fluxscaling
        if seed == -1:
            seed = np.random.randint(
                0, 1e9
            )  # Use one seed, so all regions have identical column non-uniformity.
        np.random.seed(seed)

        self.log = logging.getLogger(__file__)

    def set_ifiles(self):

        # Check if files are ASCII or FITS
        if fitslib.isfits(self.cr_length_file):
            try:
                data = pyfits.getdata(self.cr_length_file, extname="DATA")
            except KeyError:
                self.log.error(
                    "No DATA extention in FITS file : %s " % self.cr_length_file
                )
                sys.exit(1)
            crLengths = np.array(data.tolist())
        else:
            crLengths = np.loadtxt(self.cr_length_file)

        if fitslib.isfits(self.cr_distance_file):
            try:
                data = pyfits.getdata(self.cr_distance_file, extname="DATA")
            except KeyError:
                self.log.error(
                    "No DATA extention in FITS file : %s " % self.cr_distance_file
                )
                sys.exit(1)
            crDists = np.array(data.tolist())
        else:
            crDists = np.loadtxt(self.cr_distance_file)

        # Read CR informations (distribution functions)
        self.cr = dict(
            cr_u=crLengths[:, 0],
            cr_cdf=crLengths[:, 1],
            cr_cdfn=np.shape(crLengths)[0],
            cr_v=crDists[:, 0],
            cr_cde=crDists[:, 1],
            cr_cden=np.shape(crDists)[0],
        )

    def cosmicRayIntercepts(self, lum, x0, y0, l, phi, fscale):
        """
        Derive cosmic ray streak intercept points.

        :param lum: luminosities of the cosmic ray tracks
        :param x0: central positions of the cosmic ray tracks in x-direction
        :param y0: central positions of the cosmic ray tracks in y-direction
        :param l: lengths of the cosmic ray tracks
        :param phi: orientation angles of the cosmic ray tracks

        :return: map
        :rtype: nd-arrays
        """
        # create empty arrays
        crImage = np.zeros((self.ny, self.nx), dtype=np.float64)

        # x and y shifts
        dx = l * np.cos(phi) / 2.0  # beware! 0<phi< pi, dx < 0
        dy = l * np.sin(phi) / 2.0
        mskdx = np.abs(dx) < 1e-8
        mskdy = np.abs(dy) < 1e-8
        dx[mskdx] = 0.0
        dy[mskdy] = 0.0

        # pixels in x-direction
        ilo = np.round(x0.copy() - dx)
        msk = ilo < 0.0
        ilo[msk] = 0
        ilo = ilo.astype(np.int)

        ihi = np.round(x0.copy() + dx)
        msk = ihi > self.nx
        ihi[msk] = self.nx
        ihi = ihi.astype(np.int)

        # pixels in y-directions
        jlo = np.round(y0.copy() - dy)
        msk = jlo < 0.0
        jlo[msk] = 0
        jlo = jlo.astype(np.int)

        jhi = np.round(y0.copy() + dy)
        msk = jhi > self.ny
        jhi[msk] = self.ny
        jhi = jhi.astype(np.int)

        offending_delta = 1.0

        # loop over the individual events
        for i, luminosity in enumerate(lum):
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
                crImage[yc, xc] += luminosity * fscale

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

                if 0 <= cx < self.nx and 0 <= cy < self.ny:
                    crImage[cy, cx] += w * luminosity * fscale

        return crImage

    def _drawSingleEvent(self, limit=1000, cr_n=1):
        """ Generate a bunch of cosmic ray events and return a cosmic ray map
        limit : limiting energy for the cosmic ray event
        cr_n : number of events to include

        TODO : Decide if needed !
        """

        # random variable for each event (for cosmic length)
        luck = np.random.rand(int(np.floor(cr_n)))

        # draw the length of the tracks
        ius = InterpolatedUnivariateSpline(self.cr["cr_cdf"], self.cr["cr_u"])
        self.cr["cr_l"] = ius(luck)

        # set the energy directly to the limit
        self.cr["cr_e"] = np.asarray([limit] * cr_n)

        # Choose the properties such as positions and an angle from a random Uniform dist
        cr_x = self.nx * np.random.rand(int(np.floor(cr_n)))
        cr_y = self.ny * np.random.rand(int(np.floor(cr_n)))
        cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

        # find the intercepts
        CCD_cr = self.cosmicRayIntercepts(
            self.cr["cr_e"], cr_x, cr_y, self.cr["cr_l"], cr_phi, self.cr_fluxscaling
        )

        # count the covering factor
        area_cr = np.count_nonzero(CCD_cr)
        self.log.info(
            "The cosmic ray covering factor is %i pixels i.e. %.3f per cent"
            % (area_cr, 100.0 * area_cr / (self.nx * self.ny))
        )

        return CCD_cr

    def drawEventsToCoveringFactor(self, coveringFraction=1.4, limit=1000, nx=0, ny=0):
        """ Generate a bunch of cosmic ray events and return a cosmic ray map
            :param limit: limiting energy for the cosmic ray event
            :param coveringFraction : covering fraction of cr over the total number of px (in percent) normalixzed for a 565s
                                exposure time
            :param nx: x size of the map
            :param ny: y size of the map
            :return map, length, energy: map of cosmic rays, list of CR length, list of CR energies
        """

        if nx != 0:
            self.nx = nx
        if ny != 0:
            self.ny = ny

        # Prepare the CR map
        CCD_cr = np.zeros((self.ny, self.nx))

        # how many events to draw at once, too large number leads to exceeding the covering fraction

        cdf = self.cr["cr_cdf"]
        ucr = self.cr["cr_u"]
        aproxpdf = (cdf[1:] - cdf[0:-1]) / (ucr[1:] - ucr[0:-1])
        averlength = (aproxpdf * ucr[1:]).sum() / aproxpdf.sum()
        cr_tot_guess = coveringFraction / 100.0 * self.nx * self.ny / averlength

        # allocating for a max. 5% error in cover. fraction, aprox.
        # Notice that the minimum number of events will be one...
        cr_n = max(int(cr_tot_guess * 0.05), 1)

        covering = 0.0
        lengths = []
        energies = []
        cr_tot = 0

        while covering < coveringFraction:

            # pseudo-random numbers taken from a uniform distribution between 0 and 1
            luck = np.random.rand(cr_n)

            # draw the length of the tracks
            ius = interp1d(self.cr["cr_cdf"], self.cr["cr_u"], kind="slinear")
            self.cr["cr_l"] = ius(luck)

            if limit is None:
                ius = interp1d(self.cr["cr_cde"], self.cr["cr_v"], kind="slinear")
                self.cr["cr_e"] = ius(luck)
            else:
                # set the energy directly to the limit
                self.cr["cr_e"] = np.asarray([limit])

            lengths += self.cr["cr_l"].tolist()
            energies += self.cr["cr_e"].tolist()

            # Choose the properties such as positions and an angle from a random Uniform dist
            cr_x = self.nx * np.random.rand(int(np.floor(cr_n)))
            cr_y = self.ny * np.random.rand(int(np.floor(cr_n)))
            cr_phi = np.pi * np.random.rand(int(np.floor(cr_n)))

            # find the intercepts
            CCD_cr += self.cosmicRayIntercepts(
                self.cr["cr_e"],
                cr_x,
                cr_y,
                self.cr["cr_l"],
                cr_phi,
                self.cr_fluxscaling,
            )

            # count the covering factor
            area_cr = np.count_nonzero(CCD_cr)
            covering = 100.0 * area_cr / (self.nx * self.ny)

            cr_tot += cr_n
            text = (
                "The cosmic ray covering factor is %i pixels i.e. %.3f per cent (total number of cr : %i)"
                % (area_cr, covering, cr_tot)
            )
            self.log.info(text)

        return CCD_cr
