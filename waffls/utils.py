'''
Miscellaneous utilities
'''

# utils.py
# author: Ben DeVries
# email: bdv@umd.edu

import numpy as np
import math

def getbit(x, pos):
    '''
    Retrieves value (0 or 1) from integer x at position pos
    '''
    comp = 2**pos
    y = (x & comp)//comp  
    return y


def cropToImage(img1, img2, profile1, profile2):
    '''
    Crops img1 to extent of img2 and/or extends with nodata values
    ## TODO: check cases 1 to 3 for correct broadcasting ##
    '''
    aff1 = profile1['affine']
    h1 = profile1['height']
    w1 = profile1['width']

    aff2 = profile2['affine']
    h2 = profile2['height']
    w2 = profile2['width']

    if aff1[0] != aff2[0]:
        raise ValueError("Both images must have the same resolution.")
    
    # source raster extent
    esource = [
        aff1[2],
        aff1[5] + (aff1[4] * h1),
        aff1[2] + (aff1[0] * w1),
        aff1[5]
    ]

    # target extent
    etarg = [
        aff2[2],
        aff2[5] + (aff2[4] * h2),
        aff2[2] + (aff2[0] * w2),
        aff2[5]
    ]

    # new 'cropped' array
    z = np.zeros(img2.shape, dtype = profile1['dtype'])
    z.fill(profile1['nodata'])

    # vertical 'trims'
    # if +ve, the  source's extent in that direction is beyond the target's
    upper_trim = int((esource[3] - etarg[3]) / aff1[0])
    lower_trim = int((etarg[1] - esource[1]) / aff1[0])

    # case 1: source is shifted up from target
    # write data from source starting at 0
    if upper_trim >= 0 and lower_trim < 0:
        targ_yrange = (0, z.shape[0] + lower_trim)
        source_yrange = (upper_trim, img1.shape[0])

    # case 2: source is shifted down from target
    elif upper_trim < 0 and lower_trim >= 0:
        targ_yrange = (math.abs(upper_trim), z.shape[0])
        source_yrange = (0, img1.shape[0] - lower_trim)
        
    # case 3: source is contained within target ("extend")
    elif upper_trim < 0 and lower_trim < 0:
        targ_yrange = (math.abs(upper_trim), z.shape[0] + lower_trim)
        source_yrange = (0, img1.shape[0])

    # case 3: source encompasses target ("crop")
    else:
        targ_yrange = (0, z.shape[0])
        source_yrange = (upper_trim, img1.shape[0] - lower_trim - 1)
    

    ## horizontal 'trims'
    # if +ve, the  source's extent in that direction is beyond the target's
    left_trim = int((etarg[0] - esource[0]) / aff1[0])
    right_trim = int((esource[2] - etarg[2]) / aff1[0])

    # case 1: source is shifted to the left of target
    if left_trim >= 0 and right_trim < 0:
        targ_xrange = (0, z.shape[1] + right_trim)
        source_xrange = (left_trim, img1.shape[1])

    # case 2: source is shifted to the right of target
    elif left_trim < 0 and right_trim >= 0:
        targ_xrange = (math.abs(left_trim), z.shape[1])
        source_xrange = (0, img1.shape[1] - right_trim)

    # case 3: source is contained within target ("extend")
    elif left_trim < 0 and right_trim < 0:
        targ_xrange = (math.abs(left_trim), z.shape[1] + right_trim)
        source_xrange = (0, img.shape[1])

    # case 4: source encompasses target ("crop")
    else:
        targ_xrange = (0, z.shape[1])
        source_xrange = (left_trim, img1.shape[1] - right_trim - 1)


    # broadcase from source array to z
    z[targ_yrange[0]:targ_yrange[1], targ_xrange[0]:targ_xrange[1]] = img1[source_yrange[0]:source_yrange[1], source_xrange[0]:source_xrange[1]]
    ## source shape is too large by 1 in both axes ##
    ## fixed for case 4, but not for others (need to check) ##

    # new profile
    new_profile = profile1.copy()
    new_profile.update(
        affine = profile2['affine'],
        height = profile2['height'],
        width = profile2['width']
        )

    return z, new_profile
