![GitHub release (latest by date)](https://img.shields.io/github/v/release/mafarrag/hapi)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4686056.svg)](https://doi.org/10.5281/zenodo.4686056)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MAfarrag/HAPI/master)
[![Python Versions](https://img.shields.io/pypi/pyversions/HAPI-Nile.png)](https://img.shields.io/pypi/pyversions/HAPI-Nile)
[![Documentation Status](https://readthedocs.org/projects/hapi-hm/badge/?version=latest)](https://hapi-hm.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Current build status
====================


<table><tr><td>All platforms:</td>
    <td>
      <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=12419&branchName=master">
        <img src="https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/hapi-feedstock?branchName=master">
      </a>
    </td>
  </tr>
</table>

[![Build Status](https://travis-ci.com/MAfarrag/Hapi.svg?branch=master)](https://travis-ci.com/MAfarrag/Hapi)
[![Build status](https://ci.appveyor.com/api/projects/status/rys2u0l1nbmfjuww?svg=true)](https://ci.appveyor.com/project/MAfarrag/hapi)
[![Coverage Status](https://coveralls.io/repos/github/MAfarrag/Hapi/badge.svg?branch=master)](https://coveralls.io/github/MAfarrag/Hapi?branch=master)


Current release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-hapi-green.svg)](https://anaconda.org/conda-forge/hapi) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![Downloads](https://pepy.tech/badge/hapi-nile)](https://pepy.tech/project/hapi-nile) [![Downloads](https://pepy.tech/badge/hapi-nile/month)](https://pepy.tech/project/hapi-nile) [![Downloads](https://pepy.tech/badge/hapi-nile/week)](https://pepy.tech/project/hapi-nile) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![PyPI version](https://badge.fury.io/py/HAPI-Nile.svg)](https://badge.fury.io/py/HAPI-Nile) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/hapi/badges/version.svg)](https://anaconda.org/conda-forge/hapi) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/hapi.svg)](https://anaconda.org/conda-forge/hapi) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |


![Hapi](/docs/img/Hapi4.png) ![Hapi](/docs/img/name.png)


Hapi - Hydrological library for Python 
=====================================================================
**Hapi** is an open-source Python Framework for building raster-based conceptual distributed hydrological models using HBV96 lumped 
model & Muskingum routing method at a catchment scale, Hapi gives a high degree of flexibility to all components of the model 
(spatial discretization - cell size, temporal resolution, parameterization approaches and calibration).


![1](/docs/img/Picture1.png)  ![2](/docs/img/Picture2.png)

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


Installing hapi
===============

Installing `hapi` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge hapi
```

It is possible to list all of the versions of `hapi` available on your platform with:

```
conda search hapi --channel conda-forge
```

## Install from Github
to install the last development to time you can install the library from github
```
pip install git+https://github.com/MAfarrag/HAPI
```

## pip
to install the last release you can easly use pip
```
pip install HAPI-Nile
```

Quick start
===========

```
  >>> import Hapi
```
[other code samples](https://hapi-hm.readthedocs.io/en/latest/?badge=latest)


