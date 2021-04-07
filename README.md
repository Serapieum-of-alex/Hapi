[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4662170.svg)](https://doi.org/10.5281/zenodo.4662170)
[![PyPI version](https://badge.fury.io/py/HAPI-Nile.svg)](https://badge.fury.io/py/HAPI-Nile)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MAfarrag/HAPI/master)
[![conda](https://anaconda.org/mafarrag/hapi/badges/version.svg)](https://anaconda.org/MAfarrag/hapi)
[![Build Status](https://travis-ci.org/MAfarrag/Hapi.svg?branch=master)](https://travis-ci.org/MAfarrag/Hapi)
[![Python Versions](https://img.shields.io/pypi/pyversions/HAPI-Nile.png)](https://img.shields.io/pypi/pyversions/HAPI-Nile)
[![Documentation Status](https://readthedocs.org/projects/hapi-hm/badge/?version=latest)](https://hapi-hm.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/hapi-nile)](https://pepy.tech/project/hapi-nile)
[![Downloads](https://pepy.tech/badge/hapi-nile/month)](https://pepy.tech/project/hapi-nile)
[![Downloads](https://pepy.tech/badge/hapi-nile/week)](https://pepy.tech/project/hapi-nile)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


![Hapi](/docs/img/Hapi4.png) ![Hapi](/docs/img/name.png)


Hapi - Hydrological library for Python 
=====================================================================
**Hapi** is an open-source Python Framework for building raster-based conceptual distributed hydrological models using HBV96 lumped 
model & Muskingum routing method at a catchment scale, Hapi gives a high degree of flexibility to all components of the model 
(spatial discretization - cell size, temporal resolution, parameterization approaches and calibration).

(all function has been tested and validated during a Master thesis at IHE-Delft April 2018 and the library is still under development)

Hapi 

Main Features
-------------
  - Modified version of HBV96 hydrological model (Bergström, 1992) with 15 parameters in case of considering
   snow processes, and 10 parameters without snow, in addition to 2 parameters of Muskingum routing method
  - Remote sensing module to download the meteorological inputs required for the hydrologic model simulation (ECMWF) 
  - GIS modules to enable the modeler to fully prepare the meteorological inputs and do all the preprocessing 
    needed to build the model (align rasters with the DEM), in addition to various methods to manipulate and 
    convert different forms of distributed data (rasters, NetCDF, shapefiles)
  - Sensitivity analysis module based on the concept of one-at-a-time OAT and analysis of the interaction among 
    model parameters using the Sobol concept ((Rusli et al., 2015)) and a visualization
  - Statistical module containing interpolation methods for generating distributed data from gauge data, some 
    distribution for frequency analysis and Maximum likelihood method for distribution parameter estimation.
  - Visualization module for animating the results of the distributed model, and the meteorological inputs
  - Optimization module, for calibrating the model based on the Harmony search method 

The recent version of Hapi (Hapi 1.0.1) integrates the global hydrological parameters obtained by Beck et al., (2016), 
to reduce model complexity and uncertainty of parameters.

Future work
-------------
  - Developing a regionalization method for connection model parameters with some catchment characteristics for better model calibration.
  - Developing and integrate river routing method (kinematic and diffusive wave approximation)
  - Apply the model for large scale (regional/continental) cases
  - Developing a DEM processing module for generating the river network at different DEM spatial resolutions.

References
-------------
Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles, T. R. M. & Jaap Schellekens,  and L. A. B. (2016) Global-scale regionalization of hydrologic model parameters-Supporting materials 3599–3622. doi:10.1002/2015WR018247.Received

Bergström, S. (1992) The HBV model - its structure and applications. Smhi Rh 4(4), 35.

Rusli, S. R., Yudianto, D. & Liu, J. tao. (2015) Effects of temporal variability on HBV model calibration. Water Sci. Eng. 8(4), 291–300. Elsevier Ltd. doi:10.1016/j.wse.2015.12.002


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
## conda
```
conda install -c mafarrag hapi
```
## YML file
using the environment.yml file included with hapi you can create a new environment with all the dependencies installed with the latest Hapi version
in the master branch
```
conda env create --name Hapi_env -f environment.yml
```
