# waffls
---------

**wa**ter **f**raction **f**rom **L**andsat and **S**entinel-2 imagery

----------------------------------------------------------------------------------------

waffls is a collection of algorithms for estimating sub-pixel surface water fraction using medium resolution satellite data. Reader classes for Landsat data (Collection-1 and Pre-Collection), Sentinel-2 (10m or 20m resolution SAFE format) and Harmonized Landsat Sentinel-2 (HLS; 10m S10 or 30m S30/L30) data are included.

## Installation

waffls is built on top of gdal and a number of python libraries. The rasterio and pyhdf libraries must be installed first, and can be accessed through conda. See [here](https://conda.io/docs/user-guide/install/index.html) for instructions on installing conda. To install waffls in a separate python-3 conda environment, run the following:

```bash
conda config --add channels conda-forge
conda create -n waffls gdal rasterio joblib cython
conda activate waffls
git clone https://github.com/bendv/waffls
cd waffls
python setup.py install
```

## Reference

DeVries, B., Huang, C-Q., Lang, M.W., Jones, J.W., Huang, W., Creed, I.F. and Carroll, M.L. 2017. Automated quantification of surface water inundation in wetlands using optical satellite imagery. Remote Sensing, 9(8):807.
