##-------------------------------------------------
##
## Dynamic Surface Water Extent
## Based on Jones (2015)
## implemented with cython 0.24.1
##
## Ben DeVries
## 2016-07-24
##
##-------------------------------------------------


from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int16
ctypedef np.int16_t DTYPE_t
DTYPEOUT = np.uint8
ctypedef np.uint8_t DTYPEOUT_t

@cython.boundscheck(False)
def _dswe(np.ndarray[DTYPE_t, ndim=2] B, np.ndarray[DTYPE_t, ndim=2] G, np.ndarray[DTYPE_t, ndim=2] R, np.ndarray[DTYPE_t, ndim=2] NIR, np.ndarray[DTYPE_t, ndim=2] SWIR1, np.ndarray[DTYPE_t, ndim=2] SWIR2):    
    '''
    Assigns DSWE class based on a series of decision rules
    Algorithm description: 
    	http://remotesensing.usgs.gov/ecv/document/dswe_algorithm_description.pdf
    args:
        - 6 non-thermal landsat bands (in order of increasing wavelength), from which:
            - MNDWI: normalized difference between green and SWIR1 bands
            - MBSRV: sum of green and red band reflectance
            - MBSRN: sum of NIR and SWIR1 band reflectance
            - AWESH: index from Feyisa et al. (2015)
    '''
    
    # check data types
    assert B.dtype == DTYPE and G.dtype == DTYPE and R.dtype == DTYPE and NIR.dtype == DTYPE and SWIR1.dtype == DTYPE and SWIR2.dtype == DTYPE
    
    # ranges and counters
    cdef unsigned int xmax = B.shape[0]
    cdef unsigned int ymax = B.shape[1]
    cdef unsigned int x, y
    
    # initialize label array
    cdef np.ndarray[DTYPEOUT_t, ndim=2] lab = np.zeros( [xmax, ymax], dtype = DTYPEOUT )
    
    # type temp variables
    cdef int MNDWI, MBSRV, MBSRN, AWESH, z
    
    # nodata value
    cdef int NODATAVAL = -9999

    for x in range(xmax):
        for y in range(ymax):
        
            if (SWIR1[x,y] + G[x,y] == 0):
                MNDWI = -32768
            else:
                MNDWI = int( (<float>G[x,y] - <float>SWIR1[x,y]) / (<float>G[x,y] + <float>SWIR1[x,y]) * 10000 )
            MBSRV = int( G[x,y] + R[x,y] )
            MBSRN = int( NIR[x,y] + SWIR1[x,y] )
            AWESH =  int( <float>B[x,y] + (2.5*<float>G[x,y]) - (1.5*<float>MBSRN) - (0.25*<float>SWIR2[x,y]) )
            
            z = 0
            
            ## TODO: try python module 'bitarray' to represent z
            ## ie. as an array of booleans --> [test1, test2, test3, test4, test5]
            if MNDWI > 123:
                z += 1
                
            if MBSRV > MBSRN:
                z += 10

            if AWESH > 0:
                z += 100

            if (MNDWI > -5000) & (SWIR1[x,y] < 1000) & (NIR[x,y] < 1500):
                z += 1000

            if (MNDWI > -5000) & (SWIR2[x,y] < 1000) & (NIR[x,y] < 2000):
                z += 10000
            
            ## TODO: find out if bitarray type can be used here
            if ( (B[x,y] == NODATAVAL) | (MNDWI == -32768) ):
                lab[x,y] = 255 # nodata
            elif (z < 10):
                lab[x,y] = 0 # non-water
            elif ((z >= 11001) & (z <= 11111)) | ((z >= 10111) & (z < 11000)) | (z == 1111):
                lab[x,y] = 1 # water, high confidence:
            elif ((z >= 10011) & (z <= 10110)) | ((z >= 10001) & (z <= 10010)) | ((z >= 1001) & (z <= 1110)) | ((z >= 10) & (z <= 111)):
                lab[x,y] = 2 # water, moderate confidence
            elif (z == 11000) | (z == 10000) | (z == 1000):
                lab[x,y] = 3 # partial water
            else:
                lab[x,y] = 255

    return lab

