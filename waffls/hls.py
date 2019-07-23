import os
import datetime, time
import rasterio
from scipy import ndimage
import numpy as np
import warnings
from collections import OrderedDict
from affine import Affine

from .image import Image
from .utils import getbit
from .indices import tc_coef

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False  


def get_HLSInfo(x, as_DataFrame = False):
    x = os.path.basename(x)
    info = x.split('.')
    
    # dataset (S30, L30) & sensor
    dataset = info[1]
    if dataset == 'L30':
        sensor = 'OLI'
    elif dataset in ['S30', 'S10']:
        sensor = 'MSI'
    elif dataset == 'M30':
        sensor = 'MSI/OLI'
    else:
        raise ValueError("Unrecognized dataset.")
    
    # footprint (S2 tile)
    footprint = info[2]
    
    # date
    tm = time.strptime(info[3], "%Y%j")
    date = datetime.date(tm.tm_year, tm.tm_mon, tm.tm_mday)
    
    # version
    version = ".".join( [info[4], info[5]] )
    
    sceneinfo = {
        'sceneID': x[:-4],
        'dataset': dataset,
        'sensor': sensor,
        'footprint': footprint,
        'acquisition_date': date,
        'version': version
    }

    if as_DataFrame and has_pandas:
        sceneinfo = pd.DataFrame.from_dict(sceneinfo, orient = 'index').T
        sceneinfo.index = [x[:-4]]
    elif as_DataFrame and (not has_pandas):
        warnings.warn('pandas not available. Returning a dict instead of DataFrame')
    
    return sceneinfo

class HLS(Image):
    '''
    Ingests a single HLS tile (from .hdf file)
    
    The HLS user guide is available at:
        - https://nex.nasa.gov/nex/static/media/publication/HLS.v1.0.UserGuide.pdf
        - This covers (among other things), the bit order of the S10, S30 and L30 QA bits, which are used in set_mask()
    '''
    def __init__(self, filepath):

        Image.__init__(self)

        self.filepath = os.path.abspath(filepath)
        self.header = "{0}.hdr".format(self.filepath)
        if not os.path.exists(self.header):
            warnings.warn("No header file found for {0}".format(self.filepath))
        info = get_HLSInfo(filepath)
        
        self.sceneid = info['sceneID']
        self.dataset = info['dataset']
        self.sensor = info['sensor']
        self.set_date(info['acquisition_date'])
        self.version = info['version']
        self.bandnames = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        if self.dataset == 'L30':
            self.res = 30
            self.bandindices = [ "band{0:02}".format(i) for i in range(2, 8) ]
        elif self.dataset == 'S30':
            self.res = 30
            self.bandindices = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
        elif self.dataset == 'S10':
            self.res = 10
            # S10 is actually a multi-resolution dataset
            # TODO: think about how to deal with this; for now, only 10m bands are considered
            self.bandindices = ['B02', 'B03', 'B04', 'B08']
            self.bandnames = ['B', 'G', 'R', 'NIR']
        self.footprint = info['footprint']
        self.bands = None
        self.nodatavalue = -1000
        self.dims = None
        self.qa = None
        self.mask = None
        self.outer_mask = None
        self.opened = False

        # rasterio-style profile
        with rasterio.open('HDF4_EOS:EOS_GRID:"{0}":Grid:QA'.format(self.filepath)) as src:
            self.profile = src.profile.copy()
            self.profile.update(crs = src.profile['crs']['init'])

        self.width = self.profile['width']
        self.height = self.profile['height']
        self.dtype = np.int16
        self.crs = self.profile['crs']
        
        # fix transform for S10 (temporary fix...)
        if self.dataset == 'S10':
            tmp = list(self.profile['transform'])
            tmp[0] = 10
            tmp[4] = -10
            self.profile.update(transform = Affine(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]))
        
        self.profile.update(
            driver = 'GTiff', # for writing bands to disk
            count = 1, # output one band at a time
            compress = 'lzw'
            )
        
        # bounds
        aff = self.profile['transform']
        self.xmin = aff[2]
        self.xmax = aff[2] + aff[0] * self.width
        self.ymin = aff[5] + aff[4] * self.height
        self.ymax = aff[5]
    
    def read(self, verbose = False):
        self.bands = OrderedDict()
        for k, i in enumerate(self.bandindices):
            if verbose:
                print("Opening {0} band ({1})...".format(self.bandnames[k], i))
            with rasterio.open("HDF4_EOS:EOS_GRID:{0}:Grid:{1}".format(self.filepath, i)) as src:
                self.bands[self.bandnames[k]] = src.read(1).astype(self.dtype)
        if verbose:
            print("Opening QA band...")
        with rasterio.open('HDF4_EOS:EOS_GRID:"{0}":Grid:QA'.format(self.filepath)) as src:
            self.qa = src.read(1)
        self.dims = (len(self.bandindices), self.height, self.width)
        self.profile.update(dtype = self.dtype)
        self.mask = np.zeros([self.height, self.width], dtype = np.uint8)
        self.outer_mask = np.zeros([self.height, self.width], dtype = np.uint8)
        self.outer_mask[np.where(self.bands['B'] == self.nodatavalue)] = 1
        self.opened = True
        
    def set_mask(self, **kwargs):
        '''
        Constructs a mask using the QA band. 
        Keyword Args:
            - cloud:            True (default) or False
            - cloud_shadow:     True (default) or False
            - snow_ice:         True (default) or False
            - cirrus:           True (default) or False
            - buffer:           optional buffer (number of pixels) around masked pixels to also include in mask
        '''
        
        if not self.opened:
            raise ValueError("Bands must be read first.")

        msk = np.zeros([self.height, self.width], dtype = np.uint8)

        ### outside swath ###
        msk[np.where(self.outer_mask == 1)] = 1
        
        ### cirrus ###
        if not 'cirrus' in kwargs:
            kwargs['cirrus'] = True
        if kwargs['cirrus']:
            msk += getbit(self.qa, 0)
        
        ### clouds ###
        if not 'cloud' in kwargs:
            kwargs['cloud'] = True
        if kwargs['cloud']:
            msk += getbit(self.qa, 1)
            
        ### adjacent clouds ###
        if not 'adjacent_cloud' in kwargs:
            kwargs['adjacent_cloud'] = True
        if kwargs['adjacent_cloud']:
            msk += getbit(self.qa, 2)

        ### cloud shadows ###
        if not 'cloud_shadow' in kwargs:
            kwargs['cloud_shadow'] = True
        if kwargs['cloud_shadow']:
            msk += getbit(self.qa, 3)

        ## snow/ice ###
        if not 'snow_ice' in kwargs:
            kwargs['snow_ice'] = True
        if kwargs['snow_ice']:
            msk += getbit(self.qa, 4)
            
        ### aerosol quality (OLI only) ###
        ## TODO
        
        ### spatial buffer ###
        if 'buffer' in kwargs:
            tempmsk = 1 - msk
            dist = ndimage.distance_transform_edt(tempmsk)
            msk[ np.where(dist <= kwargs['buffer']) ] = 1

        # uknown keyword warnings
        for k in kwargs:
            if not k in ['cloud', 'adjacent_cloud', 'cloud_shadow', 'snow_ice', 'cirrus', 'buffer']:
                warnings.warn("'{0}' is not a recognized keyword. Ignoring...".format(k))

        self.mask = np.zeros([self.height, self.width], dtype = np.uint8)
        self.mask[np.where(msk > 0)] = 1
        msk = None
        
    def compute_index(self, index, verbose = False):
        '''
        Addition error checking for S10 data, since only 4 bands are available at 10m resolution.
        '''
        if not isinstance(index, list):
            index = [index]
        if self.dataset == "S10":
            if not all([i in ['NDVI', 'NDWI'] for i in index]):
                raise ValueError("HLS-S10 only supports NDVI and NDWI as indices.")
        
        super(HLS, self).compute_index(index, verbose = verbose)
        
        
