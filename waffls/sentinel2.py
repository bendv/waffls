'''
Sentinel-2 reader classes
'''

import os
from datetime import datetime
import rasterio
import numpy as np
from collections import OrderedDict
from scipy import ndimage

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False  

from .image import Image
from .utils import getbit
from .indices import tc_coef
    

def get_S2Info(x, as_DataFrame = False):
    x = os.path.basename(x).replace('/', '')
    info = x.split('_')
    
    # sensor (S2A or S2B)
    sensor = info[0]
    if not sensor in ['S2A', 'S2B']:
        raise ValueError("Not a valid Sentinel-2 scene")
    
    # processing level
    processing = info[1][3:]

    # date
    date = datetime.strptime(info[2], "%Y%m%dT%H%M%S")
    
    # tile
    tile = info[5][1:]
       
    sceneinfo = {
        'sceneID': os.path.splitext(x)[0],
        'sensor': sensor,
        'tile': tile,
        'acquisition_date': date,
        'processing_level': processing
    }

    if as_DataFrame and has_pandas:
        sceneinfo = pd.DataFrame.from_dict(sceneinfo, orient = 'index').T
        sceneinfo.index = [os.path.splitext(x)[0]]
    elif as_DataFrame and (not has_pandas):
        warnings.warn('pandas not available. Returning a dict instead of DataFrame')
    
    return sceneinfo


    
class S2(Image):
    '''
    Ingests a single Sentinel-2 image in SAFE format  
    '''
    def __init__(self, filepath, resolution = 20, nir_broad_band = False):
        
        if not os.path.exists(filepath):
            raise ValueError("{0} not found.".format(filepath))

        if not os.path.splitext(filepath)[1] == '.SAFE':
            raise ValueError("Only SAFE formatted S2 data are supported.")
        
        if not resolution in [10,20]:
            raise ValueError("Only 10m and 20m data are supported")
    
        Image.__init__(self)
        
        self.filepath = filepath
        info = get_S2Info(filepath)
        
        self.sceneid = info['sceneID']
        self.sensor = info['sensor']
        self.tile = info['tile']
        self.set_date(info['acquisition_date'])
        self.res = resolution
        if resolution == 10:
            self.dataset = 'S10'
            self.bandnames = ['B', 'G', 'R', 'NIR']
            self.bandindices = ['B02', 'B03', 'B04', 'B08']
        else:
            self.dataset = 'S20'
            self.bandnames = ['B', 'G', 'R', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']
            self.bandindices = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'] ##
            if nir_broad_band:
                self.bandindices[6] = 'B08'
                
        # find appropriate jp2 files
        ### TODO: figure out if handling multiple granules is necessary ###
        imgdir = []
        qadir = []
        for path, dirs, files in os.walk(self.filepath):
            for d in dirs:
                if d == 'IMG_DATA':
                    imgdir.append("{0}/R{1}m".format(os.path.join(path, d), self.res))
                    qadir.append("{0}/R20m".format(os.path.join(path, d)))
                    
        if len(imgdir) == 0:
            raise ValueError("No IMG_DIR found in scene directory")
        elif len(imgdir) > 1:
            warnings.warn("More than one granule found; using the first one only (this is a temporary workaround...)")
        imgdir = imgdir[0]
        qadir = qadir[0]
        if not os.path.exists(imgdir):
            raise ValueError("No valid image directory found for given scene and resolution")
        
        datestring = datetime.strftime(self.date, "%Y%m%dT%H%M%S")
        self.bandfiles = ["{0}/T{1}_{2}_{3}_{4}m.jp2".format(imgdir, self.tile, datestring, b, self.res) for b in self.bandindices]
        if not all([os.path.exists(b) for b in self.bandfiles]):
            raise ValueError("Some or all band files missing from IMG_DIR for given resolution")
        self.qafile = "{0}/T{1}_{2}_SCL_20m.jp2".format(qadir, self.tile, datestring)
        if not os.path.exists(self.qafile):
            raise ValueError("QA file not found in IMG_DIR")
        
        self.bands = None
        self.dims = None
        self.qa = None
        self.mask = None
        self.outer_mask = None
        self.opened = False
        
        # rasterio-style profile info
        with rasterio.open(self.bandfiles[0]) as src:
            self.profile = src.profile.copy()
        self.profile.update(nodata = 0, dtype = np.int16)
        self.width = self.profile['width']
        self.height = self.profile['height']
        self.dtype = self.profile['dtype']
        self.nodatavalue = self.profile['nodata']
        self.crs = self.profile['crs']
    
    def read(self, verbose = False):
        self.bands = OrderedDict()
        for i in range(len(self.bandnames)):
            if verbose:
                print("Opening {0} band ({1})...".format(self.bandnames[i], self.bandindices[i]))
            with rasterio.open(self.bandfiles[i]) as src:
                self.bands[self.bandnames[i]] = src.read(1).astype(self.dtype)
        
        if verbose:
            print("Opening QA (SCL) band...")
        with rasterio.open(self.qafile) as src:
            self.qa = src.read(1)
        
        # resample if needed
        if self.res == 10:
            self.qa = np.kron(self.qa, np.ones((2,2), dtype = self.dtype))
            
        self.opened = True
    
    def set_mask(self, maskvalues = [0, 1, 3, 7, 8, 9, 10, 11], buffer = None):
        '''
        QA information (SLC layer): https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm:
            0 - no data
            1 - saturated or defective
            2 - dark area pixels
            3 - cloud shadows
            4 - vegetation
            5 - not vegetated
            6 - water
            7 - unclassified
            8 - cloud (medium probability)
            9 - cloud (high probability)
            10 - thin cirrus
            11 - snow
        '''
        self.mask = np.isin(self.qa, maskvalues).astype(np.uint8)
        
        if buffer:
            tempmsk = 1 - self.mask
            dist = ndimage.distance_transform_edt(tempmsk)
            self.mask[np.where(dist <= buffer)] = 1
    
    
    
    