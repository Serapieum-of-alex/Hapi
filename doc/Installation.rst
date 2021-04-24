Installation
============


Please install Hapi in a Virtual environment so that its requirements don't tamper with your system's python
``Hapi`` works with all Python versions

conda
**********************
the easiest way to install ``Hapi`` is using ``conda`` package manager. ``Hapi`` is available in the `conda-forge <https://conda-forge.org/>`_ channel. To install
you can use the following command: 

+ ``conda install -c conda-forge hapi``

If this works it will install wflow with all dependencies including Python and gdal,
and you skip the rest of the installation instructions.


Installing Python and gdal dependencies
===========================================

The main dependencies for Hapi are an installation of Python 2.7+, and gdal

Installing Python
**********************

For Python we recommend using the Anaconda Distribution for Python 3, which is available
for download from https://www.anaconda.com/download/. The installer gives the option to
add ``python`` to your ``PATH`` environment variable. We will assume in the instructions
below that it is available in the path, such that ``python``, ``pip``, and ``conda`` are
all available from the command line.

Note that there is no hard requirement specifically for Anaconda's Python, but often it
makes installation of required dependencies easier using the conda package manager.

Install as a conda environment
==============================

The easiest and most robust way to install wflow is by installing it in a separate
conda environment. In the root repository directory there is an ``environment.yml`` file.
This file lists all dependencies. Either use the ``environment.yml`` file from the master branch
(please note that the master branch can change rapidly and break functionality without warning),
or from one of the releases {release}.

Run this command to start installing all wflow dependencies:

+ ``conda env create -f environment.yml``

This creates a new environment with the name ``hapi``. To activate this environment in
a session, run:

+ ``activate hapi``

For the installation of Hapi there are two options (from the Python Package Index (PyPI)
or from Github). To install a release of wflow from the PyPI (available from release 2018.1):

+ ``pip install hapi-nile=={release}``

To install directly from GitHub (from the HEAD of the master branch):

+ ``pip install git+https://github.com/MAfarrag/HAPI.git``

or from Github from a specific release:

+ ``pip install git+https://github.com/MAfarrag/Hapi.git@{release}``

Now you should be able to start this environment's Python with ``python``, try
``import Hapi`` to see if the package is installed.


More details on how to work with conda environments can be found here:
https://conda.io/docs/user-guide/tasks/manage-environments.html


If you are planning to make changes and contribute to the development of wflow, it is
best to make a git clone of the repository, and do a editable install in the location
of you clone. This will not move a copy to your Python installation directory, but
instead create a link in your Python installation pointing to the folder you installed
it from, such that any changes you make there are directly reflected in your install.

+ ``git clone https://github.com/MAfarrag/Hapi.git``
+ ``cd Hapi``
+ ``activate Hapi``
+ ``pip install -e .``

Alternatively, if you want to avoid using ``git`` and simply want to test the latest
version from the ``master`` branch, you can replace the first line with downloading
a zip archive from GitHub: https://github.com/MAfarrag/HAPI/archive/master.zip

Install using pip
=================

Besides the recommended conda environment setup described above, you can also install
wflow with ``pip``. For the more difficult to install Python dependencies, it is best to
use the conda package manager:

+ ``conda install numpy scipy gdal netcdf4 pyproj``


Then install a release {release} of wflow (available from release 2018.1) with pip:

+ ``pip install hapi-nile=={release}``


Check if the installation is successful
=======================================

To check it the install is successful, go to the examples directory and run the following command:

+ ``python -m Hapi.*******``

This should run without errors.


.. note::

      This documentation was generated on |today|

      Documentation for the development version:
      https://wflow.readthedocs.org/en/latest/

      Documentation for the stable version:
      https://wflow.readthedocs.org/en/stable/


TODO
====

.. todolist::
	- add the test
if you are using conda
``
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

``

Install the dependencies
**********************

you can check [libraries.io](https://libraries.io/github/MAfarrag/HAPI) to check versions of the libraries
``
conda install Numpy
conda install pandas
conda install scipy
conda install fiona
conda install -c conda-forge gdal
conda install -c conda-forge rasterio
conda install shapely
conda install geopandas
``

Install from Github
**********************

to install the last development to time you can install the library from github
``
pip install git+https://github.com/MAfarrag/HAPI
``

Compile
**********************
You can compile the repository after you clone it 
iF python is already added to your system environment variable
``
python setup.py install
# 
pathto_your_env\python setup.py install
``

pip
**********************
to install the last release you can easly use pip
``
pip install HAPI-Nile
``



YML file
**********************
using the environment.yml file included with hapi you can create a new environment with all the dependencies installed with the latest Hapi version
in the master branch
```
conda env create --name Hapi_env -f environment.yml
``
