'''
Miscellaneous utilities
'''

# utils.py
# author: Ben DeVries
# email: bdv@umd.edu

from __future__ import print_function, division, absolute_import
import numpy as np
import math

def getbit(x, pos):
    '''
    Retrieves value (0 or 1) from integer x at position pos
    '''
    comp = 2**pos
    y = (x & comp)//comp  
    return y
