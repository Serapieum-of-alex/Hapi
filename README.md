[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2572035.svg)](https://doi.org/10.5281/zenodo.2572035)
[![PyPI version](https://badge.fury.io/py/HAPI-Nile.svg)](https://badge.fury.io/py/HAPI-Nile)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MAfarrag/HAPI/master)

HAPI - Hydrological library for Python 
=====================================================================
**HAPI** is a Python package providing fast and flexible, way to build Hydrological models with different spatial representations (lumped, semidistributed and conceptual distributed) using HBV96.
The package is very flexible to an extent that it allows developers to change the structure of the defined conceptual model or to enter
their own model, it contains two routing functions muskingum cunge, and MAXBAS triangular function.
(all function has been tested and validated during a Master thesis at IHE-Delft April 2018 and the library is still under development)


Main Features
-------------
Here are just a few of the things that Hapi does well:
  - Easy handling of rasters data downloaded from global data and easy way to
    manipulate the data to arrange it to run the model
  - Easy calibration of the model using Harmony search method and Genetic Algorithms
  - flexible GIS function to process rasters interpolate values and georeference 
    calculated discharge values to the correct place.
  - Different ways for considering variability of catchment properties, lumped properties,
    distributed properties and hydrologic response units
  - Two different Ways of generating input data, from satellite data and from simple statistics
    methods like IDW(inverse distance weighting method) and ISDW(inverse dsquared istance weighting method)
  - Wide range of GIS function to process Rasters (execute some functions on a folder of rasters) like 
    project raster, resampling, Clipping, creating a raster fom arrays, Map algebra.
  - Some function to plot shapefiles using and rasters using bokeh library in a web application.
  - Different performance criteria to measure the godness of fit of the hydrological model
  
-------------
This work has been done under supervision of Dr/ Gerald Corzo at IHE Delft April 2018
-------------




Installation
============
```
Please install Hapi in a Virtual environment so that its requirements don't tamper with your system's python
**Hapi** works with Python 2.7 and 3.7 64Bit on Windows
```

if you are using conda
```
# Clone the repository. Or download the ZIP and add `-master` to the name.
git clone https://github.com/MAfarrag/HAPI

# Enter the repository
open comand prompt and type cd then the path to the repository
cd Hapi

# Create a virtualenv
open anaconda prompt and type
conda create -n Hapi_env python=3.7 anaconda 
that will create an environment named python3 that contain s Python3.7 version of Anaconda 

# Activate the env
conda activate Hapi_env

```
# Install the dependencies
you can check [libraries.io](https://libraries.io/github/MAfarrag/HAPI) to check versions of the libraries
```
conda install Numpy
conda install pandas
conda install scipy
conda install fiona
conda install -c conda-forge gdal
conda install -c conda-forge rasterio
conda install shapely
conda install geopandas
```
## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/MAfarrag/HAPI
```
## Compile 
You can compile the repository after you clone it 
iF python is already added to your system environment variable
```
python setup.py install
```
###### or 
```
pathto_your_env\python setup.py install
```
## pip
to install the last release you can easly use pip
```
pip install HAPI-Nile
```
## YML file
using the environment.yml file included with hapi you can create a new environment with all the dependencies installed with the latest Hapi version
in the master branch
```
conda env create --name Hapi_env -f environment.yml
```