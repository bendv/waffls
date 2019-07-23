''' 
Classes and methods for the pseudocluster sampling method
'''

## pseudocluster.py
## author: Ben DeVries
## email: bdv@umd.edu

from __future__ import division, absolute_import, print_function
import rasterio
import numpy as np
import math
import warnings
from collections import OrderedDict

try:
    from multiprocessing import Pool, Process
    has_multiprocessing = True
except ImportError:
    warnings.warn('Multiprocessing not available. Install multiprocessing module to enable it.')
    has_multiprocessing = False

from .image import Image
from .indices import calc_indices, _allowed_indices
from ._pseudocluster import _pseudocluster

def _in2d(x, vals):
    '''
    Wrapper for numpy.in1d().reshape(). See https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html for more info.
    '''
    test = np.in1d(x.ravel(), vals).reshape(x.shape)
    return test

def _bi(x, water_values, land_values):
    '''
    Balance Index (BI)
    (see compute_samples() method for definition)
    '''
    bi = 1 - ( abs(np.sum(_in2d(x, water_values)) - np.sum(_in2d(x, land_values))) / (np.sum(_in2d(x, water_values)) + np.sum(_in2d(x, land_values))) )
    return bi


class _PseudoclusterConfig(object):
    def __init__(self, indices = None, exclude_bands = None):
        '''
        Configuration class for Pseudocluster Class
        '''        
        pass

_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
_S10bands = ['B', 'G', 'R', 'NIR']
_S20bands = ['B', 'G', 'R', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']

class Pseudocluster(Image):
    '''
    Pseudocluster algorithm class
    Args:
        - Dataset: object of class Landsat or HLS
        - blockshape: shape of sample blocks as a tuple (height, width)
        - indices: indices to be computed
        - exclude_bands - original bands to be excluded from model
        - copy - create copy of the input Dataset class?
        - copy: create a copy of the image bands and mask in memory?
    ''' 
    def __init__(self, Dataset, blockshape, indices = None, exclude_bands = None, copy = False):
        
        if Dataset.dataset == 'S10':
            default_bands = _S10bands
            if not all([i in ['NDVI', 'NDWI'] for i in indices]):
                raise ValueError("HLS-S10 only supports NDVI and NDWI as indices.")
        elif Dataset.dataset == 'S20':
            default_bands = _S20bands
        else:
            default_bands = _bands
        
        if indices:
            if not isinstance(indices, list):
                indices = [indices]
            for i in indices:
                if not i in _allowed_indices:
                    raise ValueError("{0} is not an allowed index.".format(i))
            indices = default_bands + [i for i in indices]
        else:
            indices = default_bands

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

        Image.__init__(self)
        self.set_date(Dataset.date)

        if not Dataset.opened:
            raise ValueError("Dataset must be read first.")
        elif Dataset.mask is None:
            raise ValueError("Dataset mask must be set first.")
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

        if not len(blockshape) is 2:
            raise ValueError("blockshape must be of length 2")

        self.shape = blockshape
        self.cols = int( self.width / self.shape[0] )
        self.rows = int( self.height / self.shape[1] )
        self.tops = [ top for top in range(0, self.shape[0]*self.rows, self.shape[0]) ]
        self.lefts = [ left for left in range(0, self.shape[1]*self.cols, self.shape[1]) ]
        self.water_mask = None
        self.water_values = None
        self.land_values = None
        self.bi = None
        self.weights = None
        self.nsamples = None
        self.copy = copy
    
    def set_water_mask(self, water_mask, water_values = [1], land_values = [0]):
        '''
        Set the water map and indicate the values representing land and (pure) water
        '''
        self.water_mask = water_mask
        self.water_values = water_values
        self.land_values = land_values
    
    def block_extent(self, block_index):
        '''
        Return the extent of a given block
        block_index: zero-indexed tuple of length two (row_index, col_index)
        Value: (top, bottom, left, right)
        '''
        if (block_index[0] > self.rows) or (block_index[1] > self.cols):
            raise ValueError("block_index must be within rows and cols.")
        i, j = block_index
        top = self.tops[i]
        bottom = self.tops[i] + self.shape[0]
        left = self.lefts[j]
        right = self.lefts[j] + self.shape[1]
        return top, bottom, left, right

    def compute_indices(self):
        # wrapper for self.compute_index
        self.compute_index([i for i in self.index_names if i not in self.bandnames])
        for i in self.index_names:
            self.indices[i] = self.bands[i]

    def crop_array(self, x, block_index):
        '''
        Same as crop_bands, but on any np.array
        '''
        if len(x.shape) == 2:
            dims = x.shape
        elif len(x.shape) == 3:
            dims = (x.shape[1], x.shape[2])
        else:
            raise ValueError("Array must have 2 or 3 dimensions")

        assert dims[0] == self.height
        assert dims[1] == self.width
        top, bottom, left, right = self.block_extent(block_index)
        if len(x.shape) == 3: # TODO: force 3-D array
            z = x[:, top:bottom, left:right]
        else:
            z = x[top:bottom, left:right]
        return z

    def crop_indices(self, block_index):
        '''
        Wrapper for crop_array(), but applied only to indices
        '''
        z = np.stack([j for i,j in self.indices.items()])
        return self.crop_array(z, block_index)
    
    def compute_samples(self, d, N, minsample = 0):
        '''
        Computes sample weights (w) and # samples (n) based on the balance index (BI), dampening factor (d) and total sample size (N)
        .. math::
            BI = 1 - \frac{|N_{w}-N_{l}|}{N_{w}+N_{l}}  
            w_{ij} = \frac{BI_{ij}+d}{\sum_{ij}{(BI_{ij}+d)}};
            d \ge 0  
            n_{ij} = w_{ij}N
        
        Args:
            water_map: a 2-D numpy array with identical height and width as image data
            d: dampening factor
            N: total number of samples

            minsample: minimum sample required in a block for it to be included (otherwise a numpy.nan is assigned to weights array)
        '''
        if self.water_mask is None:
            raise ValueError("water_mask must be set first.")
            
        self.bi = np.zeros( (self.rows, self.cols), dtype = np.float32 )
        
        for i in range(self.rows):
            for j in range(self.cols):
                wc = self.crop_array(self.water_mask, (i, j))
                if np.sum( _in2d(wc, self.water_values + self.land_values) ) >= minsample:
                    self.bi[i,j] = _bi(wc, self.water_values, self.land_values)
                else: 
                    self.bi[i,j] = np.nan 

        self.weights = (self.bi + d) / np.nansum(self.bi + d)
        self.nsamples = np.round(N * self.weights).astype(np.uint32)
        self.nsamples[np.isnan(self.weights)] = 0

    def get_pseudoclusters(self, cl, ncpus=1):
        '''
        (wrapper for _runpseudocluster)
        Args:
            cl: number of pixels per pseudocluster
            ncpus: number of cpus (for parallel processing)
            
        Returns:
            A tuple of (1) a numpy.ndarray of the covariates and (2) a 1-D array of the response variable
        '''
        if self.nsamples is None:
            raise ValueError("# of samples per block must be computed first.")

        block_data = []
        z = self.stack_bands()
        for i in range(self.rows):
            for j in range(self.cols):
                block_data.append({
                    'x': self.crop_array(z, (i, j)),
                    'wm': self.crop_array(self.water_mask, (i, j)),
                    'N': self.nsamples[i, j],
                    'cl': cl
                    })

        if has_multiprocessing and (ncpus > 1):
            try:       
                p = Pool(ncpus)
                samples = p.map(_runpseudocluster, block_data)
            finally:
                p.close()
                p.join()
        else:
            samples = [ _runpseudocluster(t) for t in block_data ]
        if all(s is None for s in samples):
            return None, None
        else:
            samples = np.concatenate([ s for s in samples if s is not None ], axis = 1)
            return samples[:-1], samples[-1]

    # some other utility methods not currently used in algorithm...
    def count_obs(self, block_index):
        '''
        TODO:
        Returns a count of the valid (non-NA) pixels available in block (i, j)
        Uses the first band in Image as a reference
        Args:
            - block_index: (i, j) --> (row, column)
        '''
        pass

def samples_to_dict(samples, band_names = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']):
    '''
    Converts numpy.ndarray from get_pseudoclusters() above to OrderedDict in order to compute indices from these (see waffls.indices.calc_indices())
    Args:
        - samples: a 2-D array of shape (nbands, nsamples). If nbands > len(band_names), only the first len(band_names) bands will be taken. It is recommended just to use the 6 original spectral bands, and then compute the indices from these.
        
   
    '''
    samples_dict = OrderedDict()
    for i, b in enumerate(band_names):
        samples_dict[b] = samples[i].reshape((1, samples[i].shape[0]))
    
    return samples_dict
    

def _runpseudocluster(kwargs):
    '''
    Wrapper for pseudocluster function
    '''
    N = kwargs['N']
    if N > 0:
        x = kwargs['x']
        nbands = x.shape[0]
        npixels = x.shape[1] * x.shape[2]
        x = x.reshape([nbands, npixels])
        wm = kwargs['wm'].flatten()
        cl = kwargs['cl']
        try:
            ps = _pseudocluster(x, wm, N, cl)
        except:
            ps = None 
            # TODO: find some way to flag these errors
            # ie. in which blocks are these errors being thrown?
            # and are biases being introduced by skipping over error-prone sample blocks?
    else:
        ps = None

    return ps
