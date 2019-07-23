'''
Contains Image container classes for Landsat data
'''

## landsat.py
## author: Ben DeVries
## email: bdv@umd.edu

from __future__ import division, print_function
import rasterio
import numpy as np
from scipy import ndimage
import os
import time, datetime
import warnings
import glob
import re
from collections import OrderedDict

from .image import Image
from .utils import getbit

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False


def _try_open(filename):
    '''
    Helper function to open files
    TODO:
    - expand this function to accommodate other file storage systems (multi-band stacks, for example?)
    '''
    try:
        y = rasterio.open(filename)
    except:
        raise ValueError("Could not open {0}".format(filename))
    return y

    
def get_LandsatInfo(x, as_DataFrame = False):
    '''
    Returns a dictionary or pandas.DataFrame with scene parameters parsed from a Landsat sceneID string

    Args:
        x - string (e.g. filename) containing the Landsat sceneID
        as_DataFrame - if True, return a single-row pandas DataFrame. Otherwise, return a dictionary (default)
    '''
    
    x = os.path.basename(x)
    info = x.split("_")
    
    # sensor
    if info[0][1] == "C":
        sensor = "OLI_TIRS"
    elif info[0][1] == "O":
        sensor = "OLI"
    #elif info[0][1] == "T":
    #    sensor = "TIRS"
    elif info[0][1] == "E":
        sensor = "ETM+"
    elif info[0][1] == "T":
        sensor = "TM"
    elif info[0][1] == "M":
        sensor = "MSS"
    else:
        raise ValueError("Sensor \'{0}\' not recognized. Check sceneid.")
    
    # processing level
    proc = info[1]
    
    # path/row
    path = int(info[2][0:3])
    row = int(info[2][3:6])
    
    # acquisition date
    tm = time.strptime(info[3], "%Y%m%d")
    acqdate = datetime.date(tm.tm_year, tm.tm_mon, tm.tm_mday)
    
    # SLC status (for ETM+)
    if ( (sensor == 'ETM+') & (acqdate > datetime.date(2003, 3, 31)) ):
        slc = 'off'
    else:
        slc = 'on'
    
    # processing date
    tm = time.strptime(info[4], "%Y%m%d")
    procdate = datetime.date(tm.tm_year, tm.tm_mon, tm.tm_mday)
    
    # collection number
    collno = int(info[5])
    
    # tier
    if info[6] == "RT":
        tier = "Real-Time"
    elif info[6] == "T1":
        tier = "Tier 1"
    elif info[6] == "T2":
        tier = "Tier 2"
    else:
        tier = "unknown"

    sceneinfo = {
        'sceneID': '_'.join(info[0:7]),
        'processing_level': proc,
        'sensor': sensor,
        'slc': slc,
        'path': path,
        'row': row,
        'acquisition_date': acqdate,
        'processing_date': procdate,
        'collection_number': collno,
        'tier': tier
    }

    if as_DataFrame and has_pandas:
        sceneinfo = pd.DataFrame.from_dict(sceneinfo, orient = 'index').T
        sceneinfo.index = [x]
    elif as_DataFrame and (not has_pandas):
        warnings.warn('pandas not available. Returning a dict instead of DataFrame')
    
    return sceneinfo

def get_PreCollectionLandsatInfo(x, as_DataFrame = False):
    '''
    Returns a dictionary or pandas.DataFrame with scene parameters parsed from a Pre-Collection Landsat sceneID string

    Args:
        x - string (e.g. filename) containing the Landsat sceneID
        as_DataFrame - if True, return a single-row pandas DataFrame. Otherwise, return a dictionary (default)
    '''
    p = re.compile(u"(LE7|LT5|LT4|LC8)(\d{13})")
    m = re.search(p, x)
    info = m.group()
    if info[0:3] == 'LC8':
        sensor = 'OLI'
    elif info[0:3] == 'LE7':
        sensor = 'ETM+'
    else:
        sensor = 'TM'
    tm = time.strptime(info[9:16], "%Y%j")
    date = datetime.date(tm.tm_year, tm.tm_mon, tm.tm_mday)

    if ( (sensor == 'ETM+') & (date > datetime.date(2003, 3, 31)) ):
        slc = 'off'
    else:
        slc = 'on'

    sceneinfo = {
        'sceneID': info,
        'sensor': sensor,
        'slc': slc,
        'path': int(info[3:6]),
        'row': int(info[6:9]),
        'acquisition_date': date,
        'processing_level': None,
        'processing_date': None,
        'collection_number': 0,
        'tier': 'pre-collection'
    }

    if as_DataFrame and has_pandas:
        sceneinfo = pd.DataFrame.from_dict(sceneinfo, orient = 'index').T
        sceneinfo.index = [info]
    elif as_DataFrame and (not has_pandas):
        warnings.warn('pandas not available. Returning a dict instead of DataFrame')
    
    return sceneinfo

    
class Landsat(Image):
    def __init__(self, filepath, sceneid = None, pre_collection = False):
        '''
        Args:
            filepath: name of directory or filename housing Landsat image files
            sceneid: (optional) Landsat sceneid.
            pre_collection: read the data as a pre-collection image?

        If 'sceneid' is omitted, the sceneid will be taken from the a single band filename (band2) found within 'filepath'. There should be only one, otherwise an error will be thrown.
        '''
        Image.__init__(self)

        self.filepath = os.path.abspath(filepath)

        if pre_collection:
            get_info = get_PreCollectionLandsatInfo
            self.collection = 0 # Pre-Collection identifier
        else:
            get_info = get_LandsatInfo
            self.collection = 1

        if sceneid is None:
            fl = glob.glob("{0}/*sr_band2.tif".format(filepath))
            if len(fl) > 1:
                raise ValueError("More than one file found when searching for example (band2) file. Make sure only one Landsat scene is contained in this folder")
            else:
                s = get_info(fl[0])
        else:
            s = get_info(sceneid)
        self.sceneid = s['sceneID']
        self.sensor = s['sensor']
        self.slc = s['slc']
        self.dataset = 'Landsat'
        self.set_date(s['acquisition_date'])
        if self.sensor in ['OLI', 'OLI_TIRS']:
            self.bandindices = range(2, 8)
        else:
            self.bandindices = [1, 2, 3, 4, 5, 7]
        self.bandnames = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        self.bandfiles = tuple(
            [os.path.abspath(glob.glob("{0}/*sr_band{1}.tif".format(self.filepath, b))[0]) for b in self.bandindices]
            )
        if self.collection == 1:
            self.qafile = os.path.abspath( glob.glob("{0}/*pixel_qa.tif".format(self.filepath))[0] )
            self.cfmaskfile = None
        else:
            self.qafile = None
            self.cfmaskfile = os.path.abspath( glob.glob("{0}/*cfmask.tif".format(self.filepath))[0] )
        self.footprint = "p{0:03}r{1:03}".format(s['path'], s['row'])
        self.bands = None
        self.dims = None
        self.qa = None
        self.cfmask = None
        self.mask = None
        self.opened = False

        # get some image metadata from an example image (band2)
        with rasterio.open(self.bandfiles[1]) as src:
            self.profile = src.profile # useful for writing to file

        self.crs = self.profile['crs']
        self.dtype = self.profile['dtype']
        self.nodatavalue = self.profile['nodata']
        self.width = self.profile['width']
        self.height = self.profile['height']
        
        aff = self.profile['transform']
        self.xmin = aff[2]
        self.xmax = aff[2] + aff[0] * self.width
        self.ymin = aff[5] + aff[4] * self.height
        self.ymin = aff[5]
        self.res = aff[0]

    def read(self, verbose = False, reset_mask = True):
        '''
        Reads the SR and QA/cfmask bands into memory as an OrderedDict
        '''
        profile = _try_open(self.bandfiles[0]).profile
        self.bands = OrderedDict()
        for i, f in enumerate(self.bandfiles):
            if verbose:
                print("Reading {0}".format(f))
            self.bands[self.bandnames[i]] = _try_open(f).read()[0]
        self.width = profile['width']
        self.height = profile['height']
        self.dims = (len(self.bandfiles), self.height, self.width)
        self.dtype = profile['dtype']
        self.nodatavalue = profile['nodata']

        if self.collection == 1:
            if verbose:
                print("Reading {0}".format(self.qafile))
            self.qa = _try_open(self.qafile).read(1)
        elif self.collection == 0:
            if verbose:
                print("Reading {0}".format(self.cfmaskfile))
            self.cfmask = _try_open(self.cfmaskfile).read(1)

        if self.mask is None or reset_mask:
            self.mask = np.zeros([self.height, self.width], dtype = np.uint8)

        self.opened = True

    def set_mask(self, **kwargs):
        '''
        Constructs a mask using the QA band. 

        For Collection-1 data, the following options are available:
        - radsat:           can be a number from 0 (ignore; default) to 6 (all bands). A number between 0 and 6 (e.g. 2) indicates that 
                            saturation in any of that number of bands will result in a masked pixel.
        - cloud:            True (mask clouds; default) or False (don't mask clouds)
        - cloud_shadow:     True (mask shadows; default) or False (don't mask shadows)
        - snow:             True (mask snow/ice; default) or False (don't mask snow/ice)
        - cirrus:           True (mask high-confidence cirrus; default) or False (don't mask high-confidence cirrus) (L8 only)
        - buffer:           optional buffer (number of pixels) around masked pixels to also include in mask

        It is not recommended to set any of the confidence criteria to 'low', as those tend to mask out all imaged pixels.

        For Pre-Collection data, the following options are available:
        - cloud:            True (mask clouds; default) or False (don't mask clouds)
        - cloud_shadow:     True (mask shadows; default) or False (don't mask shadows)
        - snow:         True (mask snow, default) or False (don't mask snow)
        - buffer:           optional buffer (number of pixels) around masked pixels to also include in mask
        '''

        if not self.opened:
            raise ValueError("Image bands must be read first.")

        msk = np.zeros([self.height, self.width], dtype = np.uint8)

        if self.collection == 1:
            ### outside footprint ###
            msk += getbit(self.qa, 0)

            ### radsat ###
            if not 'radsat' in kwargs:
                kwargs['radsat'] = 0
            if kwargs['radsat'] in [1, 2]:
                msk += getbit(self.qa, 4)
            elif kwargs['radsat'] in [3, 4]:
                msk += getbit(self.qa, 3)
            elif kwargs['radsat'] >= 5:
                msk += ( getbit(self.qa, 4) & getbit(self.qa, 3) )

            ### clouds ###
            if not 'cloud' in kwargs:
                kwargs['cloud'] = True
            if kwargs['cloud']:
                msk += getbit(self.qa, 5)
            
            ### cloud shadows ###
            if not 'cloud_shadow' in kwargs:
                kwargs['cloud_shadow'] = True
            if kwargs['cloud_shadow']:
                msk += getbit(self.qa, 3)

            ### snow ###
            if not 'snow' in kwargs:
                kwargs['snow'] = True
            if kwargs['snow']:
                msk += getbit(self.qa, 4)
                
            if not 'cirrus' in kwargs:
                kwargs['cirrus'] = True
            if kwargs['cirrus']:
                msk += ( getbit(self.qa, 8) & getbit(self.qa, 9) )
                    
        elif self.collection == 0:
            if 'radsat' in kwargs:
                warnings.warn("'radsat' option is not supported in Pre-Collection datasets")

            ### outside footprint ###
            msk[np.where(self.cfmask == 255)] = 1

            ### clouds ###
            if not 'cloud' in kwargs:
                kwargs['cloud'] = True
            if kwargs['cloud']:
                msk[np.where(self.cfmask == 4)] = 1

            ### cloud shadows ###
            if not 'cloud_shadow' in kwargs:
                kwargs['cloud_shadow'] = True
            if kwargs['cloud_shadow']:
                msk[np.where(self.cfmask == 2)] = 1

            ### snow ###
            if not 'snow' in kwargs:
                kwargs['snow'] = True
            if kwargs['snow']:
                msk[np.where(self.cfmask == 3)] = 1

        ### spatial buffer ###
        if 'buffer' in kwargs:
            tempmsk = 1 - msk
            dist = ndimage.distance_transform_edt(tempmsk)
            msk[ np.where(dist <= kwargs['buffer']) ] = 1

        # uknown keyword warnings
        for k in kwargs:
            if not k in ['radsat', 'cloud', 'cloud_shadow', 'snow', 'cirrus', 'buffer']:
                warnings.warn("{0} is not a recognized keyword. Ignoring...".format(k))

        self.mask = np.zeros([self.height, self.width], dtype = np.uint8)
        self.mask[np.where(msk > 0)] = 1
        msk = None

    def close(self):
        '''
        Closes the band, qa and mask arrays
        '''
        if not self.opened:
            raise ValueError("Image is already closed.")

        self.bands = None
        self.qa = None
        self.cfmask = None
        self.mask = None
        self.opened = False
###
