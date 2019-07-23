##---------------------------------------------------------------
##
## Pseudo-cluster Sampling
## Ben DeVries, bdv@umd.edu
## 2017-02-18
##
##---------------------------------------------------------------

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
import math

from libc.stdlib cimport rand, RAND_MAX, srand

# helper function for random indices
@cython.wraparound(False)
def randidx(int stop):
    cdef float randflt = <float> rand() / <float> RAND_MAX
    cdef int randi = <int> (randflt * <float> stop)
    return randi


# X - covariates (2-D)
# dswe (1-D)
# all are flattened already

@cython.boundscheck(False)
@cython.wraparound(False)
def _pseudocluster(np.ndarray[np.int16_t, ndim=2] X, np.ndarray[np.uint8_t, ndim=1] dswe, unsigned int nsamples, unsigned int cl):

    # manual bounds and type checks
    assert X.shape[1] == dswe.shape[0]
    assert cl <= dswe.shape[0]
    assert X.dtype == np.int16
    assert dswe.dtype == np.uint8
    
    # ranges and counters
    cdef unsigned int nbands = X.shape[0] # number of covariates (bands)
    cdef unsigned int npixels = dswe.shape[0] # total number of available pixels
    cdef unsigned int k, b, i, j
    
    # min and max possible swf
    cdef float swf_min, swf_max
    
    # total number of possible water (W) and land (L) pixels
    cdef unsigned int W = np.sum(dswe == 1)
    cdef unsigned int L = np.sum(dswe == 0)
    assert (L+W) >= cl, "Not enough water and land samples to derive swf"
      
    if W == 0:
        swf_max = 0.
    elif W < cl:
        swf_max = <float> W / <float> cl
    else:
        swf_max = 1.
    
    if L == 0:
        swf_min = swf_max
    else:
        swf_min = 0.
       
    # arrays of all possible water and land reflectance values
    cdef np.ndarray[np.int16_t, ndim=2] bands_w = np.zeros( [nbands, W], dtype = np.int16 )
    bands_w.fill(-9999) # to check if they're being skipped
    j = 0
    for i in range(npixels):
        if dswe[i,] == 1:
            for b in range(nbands):
                bands_w[b,j] = X[b,i]
            j += 1
    
    cdef np.ndarray[np.int16_t, ndim=2] bands_l = np.zeros( [nbands, L], dtype = np.int16 )
    bands_l.fill(-9999) # to check if they're being skipped
    j = 0
    for i in range(npixels):
        if dswe[i,] == 0:
            for b in range(nbands):
                bands_l[b,j] = X[b,i]
            j += 1
    

    # last row is the swf, all others are the covariates
    # note: everything is cast to float to preserve SWF decimal values
    # convert back outside of cython - the wrapper function will do this (as well as the reshaping)
    cdef np.ndarray[np.float32_t, ndim=2] samples = np.zeros( [nbands + 1, nsamples], dtype = np.float32 )
    
    # target swf and number of land and water pixels needed to achieve it
    cdef float targ
    cdef unsigned int nw, nl
    
    # random indices to select from bands_w and bands_l
    cdef unsigned int idx_w, idx_l
        
    # for each sample:
        # 1) define a random "target" SWF between swf_min and swf_max
        # 2) given the cluster size, cl, determine the number of water and land pixels needed to achieve this
        # 3) randomly select indices (rows) from the covariate (X) and dswe arrays and linearly mix them

    for k in range(nsamples):
        
        # determine number of water and land samples needed to achieve targ
        if swf_max == swf_min:
            targ = swf_max
        else:
            targ = ( (<float> rand() / <float> RAND_MAX) ) * (swf_max - swf_min)
        #nw = int(math.ceil(<float> cl * targ))
        nw = int(math.floor(<float> cl * targ))
        nl = cl - nw
        
        # select those samples from the arrays
        # this is sampling with replacement, so the swf_min/max conditions above are not really necessary
        # TODO: think about whether replacement is necessary or not. Is this important?

        if W > 0:
            for i in range(nw):
                randx = randidx(W)
                for b in range(nbands):
                    samples[b,k] += <float> bands_w[b,randx-1]
        if L > 0:
            for i in range(nl):
                randx = randidx(L)
                for b in range(nbands):
                    samples[b,k] += <float> bands_l[b,randx-1]
         
        # total water pixels (to be averaged in next step)
        samples[nbands,k] = <float> nw
        
        # get average
        samples[:,k] = samples[:,k] / <float> cl
            
    return samples
    