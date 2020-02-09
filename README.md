# waffls
---------

**wa**ter **f**raction **f**rom **L**andsat and **S**entinel-2 imagery

----------------------------------------------------------------------------------------

waffls is a collection of algorithms for estimating sub-pixel surface water fraction using medium resolution satellite data. Reader classes for Landsat data (Collection-1 and Pre-Collection), Sentinel-2 (10m or 20m resolution SAFE format) and Harmonized Landsat Sentinel-2 (HLS; 10m S10 or 30m S30/L30) data are included.

## Installation

`waffls` is built on top of gdal and a number of python libraries. To install its dependencies using conda:

```bash
conda config --add channels conda-forge
conda create -n waffls gdal rasterio joblib cython scipy scikit-learn
```

You can then install waffls using `pip` in your new environment:

```bash
conda activate waffls
pip install waffls
```

Or to install from source:

```bash
conda activate waffls
git clone https://github.com/bendv/waffls
cd waffls
python setup.py install
```

Check installation and version:

```bash
python -c "import waffls; print(waffls.__version__)"
```

## Examples

### Opening a Landsat image

Using a Collection-1 Landsat TM surface reflectance image:

```python
import waffls
infile = "LT050300272011042501T1-SC20190710095708" # input directory
img = waffls.Landsat(infile)
```

Various attributes are stored as object attributes:

```python
print(img.filepath)
print(img.dataset) # Landsat
print(img.date) # Acquisition date
print(img.dtype) # data type
print(img.height, img.width) # dimensions
```

...and more. For convenience, a `rasterio`-style metadata dictionary is also included:

```python
print(img.profile)
```

By default, the image data is not read into memory. Do do this, use the `read()` method:

```python
print(img.bands) # should be `None`
img.read(verbose = True)
print(img.bands) # OrderedDict of Image bands as numpy as arrays
```

To set the QA mask use the `set_mask()` method. Several optional boolean arguments can be set to apply a saturation (False by default), cloud (True by default), cloud_shadow (True by default), snow (True by default) and cirrus (True by default). Additionally, you can also mask pixels within a specified pixel buffer around the mask by setting `buffer` to an integer value (None by default). 

```python
img.set_mask()
print(img.mask) # 1 indicates mask values, 0 unmasked
```


## Reference

DeVries, B., Huang, C-Q., Lang, M.W., Jones, J.W., Huang, W., Creed, I.F. and Carroll, M.L. 2017. Automated quantification of surface water inundation in wetlands using optical satellite imagery. Remote Sensing, 9(8):807.
