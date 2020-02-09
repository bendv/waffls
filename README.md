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

## Reference

DeVries, B., Huang, C-Q., Lang, M.W., Jones, J.W., Huang, W., Creed, I.F. and Carroll, M.L. 2017. Automated quantification of surface water inundation in wetlands using optical satellite imagery. Remote Sensing, 9(8):807.
