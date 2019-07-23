'''
SWF Model functions for learning and predicting
'''

## model.py
## author: Ben DeVries
## email: bdv@umd.edu

from __future__ import print_function, division
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
import uuid
import os
import math

def model_train(train_x, train_y, **kwargs):
    ''' 
    Trains a RandomForestRegressor using training data train_x (covariates) and train_y (response)
    Also includes keyword arguments for RandomForestRegressor
    ''' 
    clf = RandomForestRegressor(**kwargs)
    clf.fit(train_x, train_y)

    return clf

def inspect_model(clf):
    ''' 
    TODO...
    ''' 
    pass

def write_model(clf, outfile):
    '''
    TODO...
    '''
    pass

def predict_swf(model, input_filelist, tempdir = '.', verbose = False, output_filename = None, linechunk = 1000):

    # pyimpute regression
    if verbose:
        print("Loading targets and running prediction...")
    target_xs, profile = _load_targets(input_filelist)
    filepath = _regress(target_xs, model, profile, tempdir, linechunk = linechunk)
        
    # fix no_data values
    if verbose:
        print("Resetting nodata values...")
    with rasterio.open(filepath) as src:
        pred_profile = src.profile
        predicted = src.read()
    with rasterio.open(input_filelist[0]) as src:
        inp_profile = src.profile
        inp = src.read()
    msk = np.where(inp == inp_profile['nodata'])
    predicted[msk] = pred_profile['nodata']
    
    if output_filename:
        if verbose:
            print("Writing output to file...")
        with rasterio.open(output_filename, "w", **pred_profile) as dst:
            dst.write(predicted)
    
    os.remove(filepath)
    
    return predicted[0]

def rescale_swf(swf, outfile, rescale = 100, dtypeout = np.uint8, nodatavalue = 255):
    '''
    TODO
    '''
    pass



# helper functions adapted from pyimpute package
def _regress(target_xs, clf, raster_info, outdir, linechunk = 1000):
    """
    Parameters
    ----------
    target_xs: Array of explanatory variables for which to predict responses
    clf: instance of a scikit-learn Regressor
    raster_info: dictionary of raster attributes with key 'affine', 'shape' and 'srs'
    outdir : output directory

    Options
    -------
    linechunk : number of lines to process per pass; reduce only if memory is constrained
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    shape = raster_info['shape']

    profile = { # for responses output
        'transform': raster_info['transform'],
        'blockxsize': shape[1],
        'height': shape[0],
        'blockysize': 1,
        'count': 1,
        'crs': raster_info['crs'],
        'driver': u'GTiff',
        'dtype': 'float64',
        'nodata': -32768,
        'tiled': False,
        'width': shape[1]
        }

    tmpfile = "{0}.tif".format(uuid.uuid4())
    response_path = os.path.join(outdir, tmpfile)
    response_ds = rasterio.open(response_path, 'w', **profile)

    # Chunky logic
    if not linechunk:
        linechunk = shape[0]
    chunks = int(math.ceil(shape[0] / float(linechunk)))

    for chunk in range(chunks):
        row = chunk * linechunk
        if row + linechunk > shape[0]:
            linechunk = shape[0] - row
        
        # in 1D space
        start = shape[1] * row
        end = start + shape[1] * linechunk
        line = target_xs[start:end, :]

        window = ((row, row + linechunk), (0, shape[1]))

        # Predict
        responses = clf.predict(line)
        responses2D = responses.reshape((linechunk, shape[1]))
        response_ds.write_band(1, responses2D, window=window)

    return response_path

def _load_targets(explanatory_rasters):
    """
    Loads a list of explanatory rasters from file and checks that their geotransform, datatype, shape and CRS are consistent.

    Parameters
    ----------
    explanatory_rasters : List of Paths to GDAL rasters containing explanatory variables

    Returns
    -------
    expl : Array of explanatory variables
    raster_info : dict of raster info
    """

    explanatory_raster_arrays = []
    trans = None
    shape = None
    crs = None
    dtype = None

    for raster in explanatory_rasters:
        with rasterio.open(raster) as src:
            ar = src.read(1)  # TODO band num?
            
            # Save or check the geotransform
            if not trans:
                trans = src.transform
            else:
                assert trans == src.transform, "Explanatory rasters have different geotransform properties."

            # Save or check the shape
            if not shape:
                shape = ar.shape
            else:
                assert shape == ar.shape, "Explanatory rasters have different shape properties."

            # Save or check the CRS
            if not crs:
                crs = src.crs
            else:
                assert crs == src.crs, "Explanatory rasters have different CRS."

            # Save or check the datatype
            if not dtype:
                dtype = src.profile['dtype']
            else:
                assert dtype == src.profile['dtype'], "Explanatory rasters have different data types."

        # Flatten in one dimension
        arf = ar.flatten()
        explanatory_raster_arrays.append(arf)

    expl = np.array(explanatory_raster_arrays).T

    raster_info = {
        'transform': trans,
        'shape': shape,
        'crs': crs,
        'dtype': dtype
    }

    return expl, raster_info
    