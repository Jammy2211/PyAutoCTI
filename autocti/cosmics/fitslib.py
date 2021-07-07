"""
@file python/ELViS_lib/fitslib.py
@date 07/13/16
@author user
"""

# Library : fits files

import astropy.io.fits as pyfits
import numpy
import numpy.ma as MA


def get_extlist_image(filename):
    """Get the list of ImageHDU extensions"""

    ext = pyfits.info(filename, output=False)
    next = len(ext)
    imlist = []
    for iext, ext in enumerate(ext):
        if ext[2] == "ImageHDU":
            imlist.append(iext)
    return next, imlist


def narray2fits(array, outname):
    """ Create a fits file from a narray """

    # try:
    hdu = pyfits.PrimaryHDU(array)
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(outname)
    # except:
    # 	return 1

    return 0


def MA2fits(MAarray, outname):
    """ Create a fits file from a Masked arrays """

    arr = numpy.array(MA.filled(MAarray, value=0))
    res = narray2fits(arr, outname)
    return res


def narrayfits_header(array, header, outname):
    """ Create a fits file from a narray with a list of header keywords """

    # Create the HDU
    hdu = pyfits.PrimaryHDU(array)
    hdulist = pyfits.HDUList([hdu])

    # Add the keywords
    for k, v in header.iteritems():
        hdulist[0].header.update(k, v)

    # Write the file
    hdulist.writeto(outname)


def isfits(file0):
    try:
        pyobj = pyfits.open(file0)
    except IOError:
        return False
    pyobj.close()
    return True
