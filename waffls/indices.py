'''
Spectral index functions (including DSWE)
'''

## indices.py
## author: Ben DeVries
## email: bdv@umd.edu

from __future__ import division, print_function
import numpy as np
from collections import OrderedDict
import math

from ._dswe import _dswe

def ndiff(band1, band2, rescale = 10000, dtype = np.int16, nodatavalue = -9999):
    '''
    Compute the normalized difference between two bands as:
    .. math::

        \frac{band1 - band2}{band1 + band2} * rescale
    '''

    #if (len(band1.shape) != 2) | (len(band2.shape) != 2):
        #raise ValueError("Input bands must be 2-dimensional")

    nd = (band1 - band2) / (band1 + band2)
    if rescale:
        nd = nd * rescale
    if dtype:
        nd = nd.astype(dtype)
    if nodatavalue:
        nd[np.where((band1 == nodatavalue) | (band2 == nodatavalue))] = nodatavalue

    return nd

def ratio(band1, band2, rescale = 10000, dtype = np.int16, nodatavalue = -9999):
    '''
    Compute the ratio between two bands as:
    .. math::

        \frac{band1}{band2} * rescale
    '''

    #if (len(band1.shape) != 2) | (len(band2.shape) != 2):
        #raise ValueError("Input bands must be 2-dimensional")

    r = band1 / band2
    if rescale:
        r = r * rescale
    if dtype:
        r = r.astype(dtype)
    if nodatavalue:
        r[np.where((band1 == nodatavalue) | (band2 == nodatavalue))] = nodatavalue

    return r

tc_coef = [
    (0.2043, 0.4158, 0.5524, 0.5741, 0.3124, .2303), #brightness
    (-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446), #greenness
    (0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109) #wetness
    ]

def tasseled_cap(bands, tc_coef = tc_coef, nodatavalue = -9999):
    '''
    Compute the tasseled cap indices: brightness, greenness, wetness

    bands - a 6-layer 3-D (images) or 2-D array (samples) or an OrderedDict with appropriate band names
    tc_coef - a list of 3 tuples, each with 6 coefficients
    '''

    if bands.shape[0] != 6:
        raise ValueError("Input array must have 6 bands")
    if len(bands.shape) == 3:
        tc = np.zeros( (3, bands.shape[1], bands.shape[2]), dtype = np.int16 )
    elif len(bands.shape) == 2:
        tc = np.zeros( (2, bands.shape[1]), dtype = np.int16 )

    for i, t in enumerate(tc_coef):
        for b in range(6):
            tc[i] += (bands[b] * t[b]).astype(np.int16)
           
        if nodatavalue:
            tc[i][ np.where(bands[0] == nodatavalue) ] = nodatavalue

    return tc 

def dswe(bands, nodatavalue = -9999):
    '''
    Compute the raw DSWE from all 6 bands
    The output is a classified product of type numpy.uint8:
        - 0: land
        - 1: water
        - 2: water (moderate confidence)
        - 3: partial water
        - 255: no_data

    bands must be given as a numpy.stack with 6 layers
    '''
    if (len(bands.shape) != 3) | (bands.shape[0] != 6):
        raise ValueError("bands must be a 3-D array with 6 layers")

    ds = _dswe(bands[0], bands[1], bands[2], bands[3], bands[4], bands[5])
    ds[np.where(bands[0] == nodatavalue)] = 255

    return ds

_allowed_indices = ['MNDWI', 'NDWI', 'NDVI', 'TCB', 'TCG', 'TCW', 'TCWGD', 'TCWGBD', 'IVR', 'IR', 'NDMI', 'COSDOY']

def calc_indices(bands, indices, tc_coef = tc_coef, doy = None, shape = None, mask = None, nodatavalue = -9999, dtype = np.int16, inplace = False, verbose = False):
    '''
    - 'bands' is an OrderedDict as defined in the Image class
    - if COSDOY is included as an index, 'doy' must be given as an integer between 1 to 366
    - if COSDOY is included, the shape will be taken from bands['B'] unless shape is explicitly given as an argument (e.g., if bands['B'] is not present)
    - if COSDOY is included, a mask must be given explicitly, otherwise no pixels will be masked
    '''

    if not isinstance(indices, list):
        indices = [indices]
    for i in indices:
        if not i in _allowed_indices:
            raise ValueError("{0} is not a supported index.".format(i))
    indices = [i.upper() for i in indices]
    if not inplace:
        bands = bands.copy()

    if any([i.startswith('TC') for i in indices]):
        z = np.stack([ bands[i] for i in ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'] ])
        tc = tasseled_cap(z, tc_coef = tc_coef, nodatavalue = nodatavalue)
        z = None
    else:
        tc = None

    for i in indices:
        if verbose:
            print("Computing {0} ...".format(i))
        if i == 'MNDWI':
            bands['MNDWI'] = ndiff(bands['G'], bands['SWIR1'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'NDWI':
            bands['NDWI'] = ndiff(bands['G'], bands['NIR'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'NDVI':
            bands['NDVI'] = ndiff(bands['NIR'], bands['R'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'NDMI':
            bands['NDMI'] = ndiff(bands['NIR'], bands['SWIR1'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'IR':
            bands['IR'] = ndiff(bands['NIR'], bands['SWIR2'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'IVR':
            bands['IVR'] = ratio(bands['SWIR1'], bands['G'], dtype = dtype, nodatavalue = nodatavalue)
        elif i == 'TCB':
            bands['TCB'] = tc[0] 
        elif i == 'TCG':
            bands['TCG'] = tc[1] 
        elif i == 'TCW':
            bands['TCW'] = tc[2]
        elif i == 'TCWGD':
            bands['TCWGD'] = tc[2] - tc[1]
            bands['TCWGD'][np.where(tc[2] == nodatavalue)] = nodatavalue
        elif i == 'TCWGBD':
            bands['TCWGBD'] = tc[2] - tc[1] - tc[0]
            bands['TCWGBD'][np.where(tc[2] == nodatavalue)] = nodatavalue
        elif i == 'COSDOY':
            if not doy:
                raise ValueError("'doy' must be set if DOY is included")
            if doy < 1 or doy > 366:
                raise ValueError("'doy' must be an integer between 1 and 366.")
            cosdoy = int( math.cos((2*np.pi * doy) / 366) * 10000 )
            if not shape:
                shape = bands['B'].shape
            bands['COSDOY'] = np.zeros(shape, dtype = dtype)
            bands['COSDOY'].fill(cosdoy)
            if mask:
                bands['COSDOY'][np.where(mask)] = nodatavalue

        if not inplace:
            return bands
