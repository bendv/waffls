# waffls
---------

**wa**ter **f**raction **f**rom **L**andsat and **S**entinel-2 imagery

----------------------------------------------------------------------------------------

waffls is a collection of algorithms for estimating sub-pixel surface water fraction using medium resolution satellite data. Reader classes for Landsat data (Collection-1 and Pre-Collection) as well as Harmonized Landsat Sentinel-2 (HLS) data are included.

## Installation

waffls is built on top of gdal and a number of python libraries. The necessary prerequisite libraries must be installed first, and can be easily accessed through conda. See [here](https://conda.io/docs/user-guide/install/index.html) for instructions on installing conda. 

To install waffls in a separate conda environment, run the following:

```bash
conda create -n waffls -c conda-forge gdal rasterio cython joblib
conda activate waffls
git clone https://github.com/bendv/waffls
cd waffls
python setup.py install
```

## References

If you use ```waffls``` in your work, please cite:

DeVries, B., Huang, C-Q., Lang, M.W., Jones, J.W., Huang, W., Creed, I.F. and Carroll, M.L. 2017. Automated quantification of surface water inundation in wetlands using optical satellite imagery. Remote Sensing, 9(8):807.

If you use the Dynamic Surface Water Extent product (ie., using ```waffls.dswe```), please cite:

Jones, J.W., 2015. Efficient wetland surface water detection and monitoring via landsat: Comparison with in situ data from the everglades depth estimation network. Remote Sensing 7, 12503–12538.

