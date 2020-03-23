[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2572035.svg)](https://doi.org/10.5281/zenodo.2572035)

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

# Install the dependencies
# there is some library that you have to install it from some conda channels
conda install Numpy
conda install pandas
conda install scipy
conda install gdal
conda install shapely
conda install geopandas
conda install fiona

# after installing all dependencies to install Hapi 
python setup.py install
# iF you have more than one environment installed in you operating system 
pathto_your_env\python setup.py install
# or in your python environment you can try to clone and install it directly from pip
pip install git+https://github.com/MAfarrag/HAP


```
