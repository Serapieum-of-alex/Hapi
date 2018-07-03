
HAPI - Hydrological library for Python
=====================================================================
**HAPI** is a Python package providing fast and flexible, way to build distributed
hydrological model using lumped conceptual model 

'''
Main Features
-------------
Here are just a few of the things that pandas does well:
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
This work has beed done as a Part of A Master Thesis in Hydroinformatics under supervision of Dr/ Gerald Corzo & Prof/ Dimitri Solomatine at IHE Delft April 2018
-------------

'''

'''
Installation
============
Please install Hapi in a Virtual environment so that its requirements don't tamper with your system's python
**Hapi** works with Python 2.7 64Bit on Windows
'''

if you are using conda
'''
# Clone the repository. Or download the ZIP and add `-master` to the name.
git clone https://github.com/MAfarrag/HAPI

# Enter the repository
cd Hapi

# Create a virtualenv
conda create -n python2 python=2.7 anaconda 
that will create an environment named python2 that contain s Python2.7 version of Anaconda 

# Activate the env
conda activate python2

# Install the dependencies
python -m pip install -r requirements.txt

'''






