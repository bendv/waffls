'''
Classes and methods for the self-trained aggregation SWF algorithm

Description
===========

Class Aggswf inherits from either Landsat or HLS depending on the value of 'platform'. Some of the original methods in these classes are disabled or modified to streamflow steps in the Aggswf algorithm.
'''

## aggswf.py
## author: Ben DeVries
## email: bdv@umd.edu

from __future__ import division, print_function
import numpy as np
from scipy import ndimage
import warnings
import os
import rasterio
from collections import OrderedDict

from .image import Image
from .indices import calc_indices, _allowed_indices

def _in2d(x, vals):
    '''
    Wrapper for numpy.in1d().reshape(). See https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html for more info.
    '''
    test = np.in1d(x.ravel(), vals).reshape(x.shape)
    return test

class _AggswfConfig(object):
    '''
    Configuration class for the aggswf algorithm
    '''
    def __init__(self, aggregate_factor = 5, initial_swf = 1., indices = None, exclude_bands = None):

        self.aggregate_factor = aggregate_factor
        self.initial_swf = initial_swf
        self.platform = None
        self.swf = None
        self.aggregated_swf = None
        self.aggregated_bands = OrderedDict()
        self.aggregated_mask = None

        # self.indices is a list of indices that will later be used to compute aggregated and unaggregated indices for sampling
        # original bands will be included in this list unless they are set in exclude_bands
        # however, any bands in exclude_bands will be kept in the 'bands' attribute in case they are needed to compute other indices
        # implying that the self.indices list will be used to (1) decide which indices to compute and (2) decide which rows of the sample array to return at the last step
        if indices:
            if not isinstance(indices, list):
                indices = [indices]
            for i in indices:
                if not i in _allowed_indices:
                    raise ValueError("{0} is not an allowed index.".format(i))
            indices = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2'] + [i for i in indices]
        else:
            indices = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        if exclude_bands:
            if not isinstance(exclude_bands, list):
                exclude_bands = [exclude_bands]
            self.index_names = [i for i in indices if not i in exclude_bands]
        else:
            self.index_names = indices
        self.indices = OrderedDict()
        self.aggregated_indices = OrderedDict()
        for i in self.index_names:
            self.indices[i] = None
            self.aggregated_indices[i] = None
            # these will be filled in later
        self.index_filenames = OrderedDict()

# Aggswf class - conditional inheritence depending on 'platform' argument
# see discussion: https://stackoverflow.com/questions/32598363/conditional-class-inheritance-in-python
# I can't seem to figure this out, so a workaround is to manually copy over the relevant attributes, which is what I've done here:
class Aggswf(_AggswfConfig, Image):
    '''
    Self-trained aggregated SWF algorithm class
    Manually copies relevant attributes from Image instance given in args
    '''
    def __init__(self, Dataset, copy = False, **kwargs):
        
        if Dataset.dataset == 'S10':
            raise ValueError("S10 data is not supported yet.")
        
        _AggswfConfig.__init__(self, **kwargs)
        Image.__init__(self)
        self.set_date(Dataset.date)

        if not Dataset.opened:
            raise ValueError("Image must be read first.")
        elif Dataset.mask is None:
            raise ValueError("Image mask must be set first.")
        else:
            self.opened = True

        if copy:
            self.bands = Dataset.bands.copy()
            self.mask = Dataset.mask.copy()
        else:
            self.bands = Dataset.bands
            self.mask = Dataset.mask
        self.sceneid = Dataset.sceneid
        self.bandnames = Dataset.bandnames
        self.dtype = Dataset.dtype
        self.nodatavalue = Dataset.nodatavalue
        self.profile = Dataset.profile
        self.height = Dataset.height
        self.width = Dataset.width
        self.nodatavalue = Dataset.nodatavalue
        self.filepath = Dataset.filepath

        self.aggregated_bands = OrderedDict()
        for i in self.bands:
            self.aggregated_bands[i] = None
        self.aggregated_mask = None
        self.swf = None
        self.aggregated_swf = None
        self.copy = copy

    def init_swf(self, water_map, land_values = [0], water_values = [1,2], partial_water_values = [3]):
        '''
        Initializes SWF values based on land, water and partial-water pixel indices based on a user-defined water_map, checking the dimensions of the map against the image data. 
        Default values are taken from the raw DSWE class definitions, 

        - water_map: a 2-D numpy array with identical height and width as image data
        - land: list of values (classes) representing land (non-water)
        - water: list of values representing (open) water
        - partial_water: list of values reprenting partial water (can be None if this class does not exist).
        '''
        dims = water_map.shape
        if len(dims) != 2:
            raise ValueError("water_map must be a 2-D numpy array.")
        if (dims[0] != self.height) | (dims[1] != self.width):
            raise ValueError("water_map must have the same width and height as the image data")

        land_idx = np.where(_in2d(water_map, land_values))
        water_idx = np.where(_in2d(water_map, water_values))
        if partial_water_values:
            partial_water_idx = np.where(_in2d(water_map, partial_water_values))

        self.swf = np.zeros(dims, dtype = np.float32)
        self.swf.fill(-32768)
        self.swf[land_idx] = 0
        self.swf[water_idx] = 1
        self.swf[partial_water_idx] = self.initial_swf
    
    def aggregate_swf(self):
        '''
        Aggregates initial SWF to coarser resolution according to self.aggregate. Also sets aggregated mask (if not already set)
        '''
        if self.swf is None:
            raise ValueError("swf must be initialized first.")

        if self.aggregated_mask is not None:
            msk = self.mask.copy().astype(np.float32)
            self.aggregated_mask = ndimage.zoom(msk, 1/self.aggregate_factor, order = 2, prefilter = False)
            self.aggregated_mask[self.aggregated_mask > 0] = 1
            self.aggregated_mask = self.aggregated_mask.astype(np.uint8)
        self.aggregated_swf = ndimage.zoom(self.swf, 1/self.aggregate_factor, order = 2, prefilter = False)
        self.aggregated_swf[ np.where(self.aggregated_mask == 1) ] = -32768
        self.aggregated_swf[ np.where(self.aggregated_swf < 0) ] = -32768 # redundant -- TODO: figure out why this is necessary!
        msk = None

    def compute_indices(self, verbose = False):
        # wrapper for self.compute_index
        self.compute_index([i for i in self.index_names if i not in self.bandnames], verbose = verbose)
        for i in self.index_names:
            self.indices[i] = self.bands[i]

    def write_indices(self, output_dir, verbose = False):
        profile = self.profile.copy()
        profile.update(
            count = 1,
            dtype = self.dtype,
            nodata = self.nodatavalue,
            compress = 'lzw'
            )
        
        for i, x in self.indices.items():
            outfl = "{0}/{1}_{2}.tif".format(output_dir, self.sceneid, i)
            if verbose:
                print("Writing {0}...".format(outfl))
            with rasterio.open(outfl, 'w', **profile) as dst:
                dst.write(x.reshape((1, self.height, self.width)))
            self.index_filenames[i] = outfl

    def delete_indices_from_disk(self):
        '''
        Removes any files written using write_bands() from disk, and resets output_filenames attribute
        '''
        if self.index_filenames is None:
            raise ValueError("No index files to delete.")

        for f in self.index_filenames:
            try:
                os.path.remove(f)
            except:
                warnings.warn("{0} not found; skipping...".format(f))
        self.index_filenames = None

    def aggregate_indices(self, verbose = True):
        if not self.opened:
            raise ValueError("bands must be opened first.")
        if any([i is None for i in self.indices]):
            raise ValueError("indices must be computed first.")

        if self.aggregated_mask is not None:
            msk = self.mask.copy().astype(np.float32)
            self.aggregated_mask = ndimage.zoom(msk, 1/self.aggregate_factor, order = 2, prefilter = False)
            self.aggregated_mask[self.aggregated_mask > 0] = 1
            self.aggregated_mask = self.aggregated_mask.astype(np.uint8)

        for i in self.bands:
            self.aggregated_bands[i] = ndimage.zoom(self.bands[i], 1/self.aggregate_factor, order = 2, prefilter = False).astype(self.dtype)
            self.aggregated_bands[i][ np.where(self.aggregated_mask == 1) ] = self.nodatavalue

        _aggregated_indices = calc_indices(
            self.aggregated_bands,
            [i for i in self.index_names if i not in self.bandnames], 
            nodatavalue = self.nodatavalue, 
            dtype = self.dtype,
            verbose = verbose
            )
        self.aggregated_indices = OrderedDict({i:j for i, j in _aggregated_indices.items() if i in self.index_names})

    def export_samples(self):
        '''
        Prepares and exports response (SWF) and covariates for input into a model (see model.py module)
        TODO: insert option for stratified random sampling
        '''
        if self.aggregated_swf is None:
            raise ValueError("swf must be set and aggregated first")
        if self.aggregated_indices is None:
            raise ValueError("indices must be computed and aggregated first")

        swf = self.aggregated_swf.flatten()
        idx = np.where(swf != -32768)
        train_y = swf[idx]
        
        covs = []
        for i, x in self.aggregated_indices.items():
            covs.append(x.flatten()[idx])
        train_x = np.vstack(covs).T

        return train_x, train_y
