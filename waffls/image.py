'''
Generic Image class with attributes inherited by Landsat and HLS classes
'''

## image.py
## Ben DeVries
## email: bdv@umd.edu

from __future__ import absolute_import, print_function, division
import numpy as np
import os
import rasterio
import warnings
from datetime import datetime, date
from collections import OrderedDict

from .indices import calc_indices, tc_coef


class Image(object):
    
    def __init__(self):
        self.bands = None
        self.qa = None
        self.cfmask = None
        self.profile = None
        self.width = None
        self.height = None
        self.dims = None
        self.mask = None
        self.opened = False
        self.output_filenames = None
        self.date = None
        
    def set_date(self, date):
        self.date = date
        
    def set_external_mask(self, mask, replace = False):
        '''
        Sets external mask
        If replace = True, the new mask will completely replace any existing mask
        If replace = False, the final mask will include pixels where either mask has a non-zero value (ie. update existing mask)
        The final mask will only consist of zeros and ones regardless of the value range of the input mask.
        '''
        if not self.opened:
            raise ValueError("Image bands must be opened first.")

        if len(mask.shape) != 2:
            raise ValueError("Input mask must be 2-dimensional.")
        elif mask.shape[0] != self.height or mask.shape[1] != self.width:
            raise ValueError("Input mask must be the same shape as image.")

        if replace or self.mask is None:
            self.mask = np.zeros((self.height, self.width), dtype = np.uint8)
            self.mask[np.where(mask != 0)] = 1
        else:
            idx = np.where((self.mask == 1) and (mask != 0))
            self.mask[idx] = 1

    def apply_mask(self):
        '''
        Set's band pixel values to self.nodata where mask != 0
        '''
        if not self.opened:
            raise ValueError("Bands must be read first.")
        if self.mask is None:
            raise ValueError("Mask must be set first.")
        for i in self.bands:
            self.bands[i][np.where(self.mask != 0)] = self.nodatavalue
    
    def reset_nodatavalue(self, new_nodatavalue):
        '''
        Replaces current nodatavalue (default HLS: -1000; Landsat: -9999) with a user specified new nodatavalue (e.g. -9999, to match Landsat) in all bands
        '''
        if not self.opened:
            raise ValueError("Image bands must be opened first.")
            
        for i in self.bands:
            self.bands[i][np.where(self.bands[i] == self.nodatavalue)] = new_nodatavalue
        
        self.nodatavalue = new_nodatavalue
        self.profile.update(nodata = self.nodatavalue)

    def compute_stats(self, stat = 'nobs'):
        '''
        Returns a dictionary of stats if stat is one of 'mean', 'median', 'sd' or a single value if stat is 'nobs'
        TODO: consider returning a table of band stats (dict or optionally a pandas.DataFrame)
        '''
        pass
        
    def compute_index(self, index, tc_coef = tc_coef, verbose = False):
        ### TODO: consider moving this outside as function to be called from various classes ###
        ### e.g. a function in indices.py that takes an OrderedDict of bands as an argument and appends the indices to it ###
        '''
        Compute one or more of the following indices and add to dataset:
            - MNDWI = (G - SWIR1) / (G + SWIR1)
            - NDWI = (G - NIR) / (G + NIR)
            - NDVI = (NIR - R) / (NIR + R)
            - NDMI = IR = (NIR - SWIR1) / (NIR + SWIR1)
            - IVR = SWIR1 / G
            - TCB: tasseled cap brightness
            - TCG: tasseled cap greenness
            - TCW: tasseled cap wetness
            - TCGWD: wetness/greenness difference
            - TCGWBD: wetness/greenness/brightness difference
            - COSDOY: time index as cos(2pi*doy/366)
        '''
        if not self.opened:
            raise ValueError("bands must be read first.")
        
        doy = int(
            datetime.strftime(datetime.combine(self.date, datetime.min.time()), "%j")
            )

        calc_indices(self.bands, index, tc_coef = tc_coef, doy = doy, dtype = self.dtype, nodatavalue = self.nodatavalue, verbose = verbose, inplace = True)

    def add_index(self, array, index_name):
        '''
        Add an already-computed index to the dataset
        *** NOT TESTED ***
        '''
        index_name = index_name.upper()

        if len(array.shape) != 2:
            raise ValueError("array should be 2-dimensional")
        if array.shape[0] != self.height or array.shape[1] != self.width:
            raise ValueError("array must be the same width and height as image.")

        if index_name in self.bands:
            raise ValueError("Index already exists: {0}".format(index_name))
        
        self.bands[index_name] = array

    def stack_bands(self, bandnames = None, copy = True):
        '''
        Returns a numpy stack of all (if bands is None) or some of the bands in the dataset.
        Copies arrays in memory if copy is True.
        '''
        z = []
        for i in self.bands:
            if not bandnames or i in bandnames:
                if copy:
                    z.append(self.bands[i].copy())
                else:
                    z.append(self.bands[i])

        return np.stack(z)
        
    def write_bands(self, output_dir, verbose = False):
        '''
        Write all bands and indices to a individual tif files in output_dir
        '''
        self.output_filenames = OrderedDict()
        profile = self.profile.copy()
        profile.update(
            count = 1,
            dtype = self.dtype,
            nodata = self.nodatavalue,
            compress = 'lzw'
            )
        
        for i in self.bands:
            outfl = "{0}/{1}_{2}.tif".format(output_dir, self.sceneid, i)
            if verbose:
                print("Writing {0}...".format(outfl))
            with rasterio.open(outfl, 'w', **profile) as dst:
                dst.write(self.bands[i].reshape((1, self.height, self.width)))
            self.output_filenames[i] = outfl

    def delete_bands_from_disk(self):
        '''
        Removes any files written using write_bands() from disk, and resets output_filenames attribute
        *** NOT TESTED ***
        '''
        if not self.output_filenames:
            raise ValueError("No band/index files to delete.")

        for f in self.output_filenames:
            try:
                os.path.remove(f)
            except:
                warnings.warn("{0} not found; skipping...".format(f))
        self.output_filenames = None

    def close(self):
        '''
        '''
        self.bands = None
        self.opened = False
        self.qa = None
        self.cfmask = None
        self.mask = None
