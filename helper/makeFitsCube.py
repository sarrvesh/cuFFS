#!/usr/bin/env python
"""
makeFitsCube.py

This script is intended for merging channel maps into a FITS cube.

Written by Sarrvesh S. Sridhar

To do list:
* Change all pyFits dependencies to AstroPy
"""
import optparse
import glob
import os
try:
    import numpy as np
except ImportError:
    raise Exception('Unable to import Numpy')
try:
    import pyfits as pf
except ImportError:
    raise Exception('Unable to import pyFits.')

version_string = 'v1.0, 8 June 2015\nWritten by Sarrvesh S. Sridhar'
print 'makeFitsCube.py', version_string
print ''

def getValidFitsList(fileList):
    """
    Extracts fits files from a list of files using its magic number.
    """
    validFitsList = []
    for name in fileList:
        if 'FITS' in os.popen('file {}'.format(name)).read():
            validFitsList.append(name)
    return validFitsList

def checkFitsShape(fitsList):
    """
    Checks if the list of fits files all have the same shape.
    If True, return the shape and memory in bytes of a single fits file
    If False, raises an exception causing the execution to terminate
    """
    for i, name in enumerate(fitsList):
        if i == 0:
            templateShape = pf.open(name, readonly=True)[0].data[0].shape
        elif templateShape != pf.open(name, readonly=True)[0].data[0].shape:
            raise Exception('Fits file {} has an incompatible shape'.format(name))
    return templateShape

def concatenateWithMemMap(validFitsList, shape, memMapName, FLAG):
    """
    Concatenate a given list of fits files into a single cube using memory map.
    Return the concatenated array and frequency list
    """
    concatCube = np.memmap(memMapName, dtype='float32', mode='w+',\
                           shape=(1, len(validFitsList), shape[-2], shape[-1]))
    freqList = []
    for i, name in enumerate(validFitsList):
        tempData = pf.open(name, readonly=True)[0].data[0]
        tempHead = pf.open(name, readonly=True)[0].header
        freqList.append(tempHead[FLAG])
        if len(shape) == 3:
            concatCube[0, i, :] = tempData[0, :]
        if len(shape) == 4:
            concatCube[0, i, :] = tempData[0, 0, :]
    return concatCube, freqList

def main(options):
    """
    Main function
    """
    memMapName = 'memMap'

    # Check user input
    if options.inp == '':
        raise Exception('An input glob string must be specified.')
    if options.out == '':
        raise Exception('An output filename must be specified.')

    # Get the list of FITS files
    fileList = sorted(glob.glob(options.inp))
    print fileList
    validFitsList = getValidFitsList(fileList)
    print 'INFO: Identified {} fits files from {} files selected by input string'.\
          format(len(validFitsList), len(fileList))

    # Proceed with the execution if we have non-zero FITS files
    if len(validFitsList) == 0:
        raise Exception('No valid fits files were selected by the glob string')

    # Check if the list of supplied fits files have the same shape
    shape = checkFitsShape(validFitsList)
    print 'INFO: All fits files have shape {}'.format(shape)
    if len(shape) not in [3, 4]:
        raise Exception('Fits files have unknown shape')
    
    # Merge the cubes
    if options.restfrq:
        FLAG = 'RESTFREQ'
    else:
        FLAG = 'CRVAL3'
    finalCube, freqList = concatenateWithMemMap(validFitsList, shape, memMapName, FLAG)
    
    # Write the frequency list to disk
    f = open(options.freq, "w")
    for line in freqList:
        f.write(str(line)+'\n')
    f.close()

    # Get a template header from a fits file
    header = pf.open(validFitsList[0], readonly=True)[0].header
    print 'INFO: Writing the concatenated fits file to {}'.format(options.out)
    hdu = pf.PrimaryHDU(data=finalCube, header=header)
    hdu.writeto(options.out)
    os.remove(memMapName)

if __name__ == '__main__':
    opt = optparse.OptionParser()
    opt.add_option('-i', '--inp', help='Glob selection string for input files '+
                   '[no default]', default='')
    opt.add_option('-o', '--out', help='Filename of the output cube '+
                   '[default: mergedFits.fits]', default='mergedFits.fits')
    opt.add_option('-f', '--freq',
                   help='Filename to write the frequency list [default: frequency.txt]',
                   default='frequency.txt')
    opt.add_option('-r', '--restfrq', help='Frequency is stored in RESTFRQ '+
                   'instead of CRVAL3 [default: False]', default=False, 
                   action='store_true')
    inOpts, arguments = opt.parse_args()
    main(inOpts)
