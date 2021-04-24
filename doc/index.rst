  .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4686056.svg
   :target: https://doi.org/10.5281/zenodo.4686056

[![PyPI version](https://badge.fury.io/py/HAPI-Nile.svg)](https://badge.fury.io/py/HAPI-Nile)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MAfarrag/HAPI/master)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/hapi/badges/version.svg)](https://anaconda.org/conda-forge/hapi)
[![Build Status](https://travis-ci.org/MAfarrag/Hapi.svg?branch=master)](https://travis-ci.org/MAfarrag/Hapi)
[![Coverage Status](https://coveralls.io/repos/github/MAfarrag/Hapi/badge.svg?branch=master)](https://coveralls.io/github/MAfarrag/Hapi?branch=master)
[![Python Versions](https://img.shields.io/pypi/pyversions/HAPI-Nile.png)](https://img.shields.io/pypi/pyversions/HAPI-Nile)
[![Documentation Status](https://readthedocs.org/projects/hapi-hm/badge/?version=latest)](https://hapi-hm.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/hapi-nile)](https://pepy.tech/project/hapi-nile)
[![Downloads](https://pepy.tech/badge/hapi-nile/month)](https://pepy.tech/project/hapi-nile)
[![Downloads](https://pepy.tech/badge/hapi-nile/week)](https://pepy.tech/project/hapi-nile)
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


Current release info
====================

  .. image:: https://img.shields.io/github/commit-activity/m/mafarrag/HAPI   :alt: GitHub commit activity
  
  .. image:: https://img.shields.io/github/issues/mafarrag/HAPI   :alt: GitHub issues

+------------+------------+-----------+------------+
|   Name     | Downloads  |  Version  |  Platforms |
+============+============+===========+============+
| body row 1 |   
.. image:: https://anaconda.org/conda-forge/hapi/badges/downloads.svg 
:target: https://anaconda.org/conda-forge/hapi| 
  .. image:: https://img.shields.io/conda/vn/conda-forge/hapi   :alt: Conda (channel only) |   
    .. image:: https://img.shields.io/gitter/room/mafarrag/Hapi   :alt: Gitter |
+------------+            +           +-----------+
| body row 2 | 
  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads :target: https://pepy.tech/project/hapi-nile  |   .. image:: https://img.shields.io/pypi/v/hapi-nile   :alt: PyPI 
  |   column 4    |
+------------+            +          +-----------+
| body row 3 | 
  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month :target: https://pepy.tech/project/hapi-nile    | column 3   | column 4|
+------------+            |  |    |
| body row 4 | 
  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=week&units=international_system&left_color=grey&right_color=blue&left_text=downloads/week :target: https://pepy.tech/project/hapi-nile | column 3  |    column 4   |
+------------+------------+-----------+-----------+



  
  .. image:: docs/img/Hapi4.png
   :width: 40pt
  
  .. image:: docs/img/name.png
   :width: 40pt


Hapi - Hydrological library for Python 
=====================================================================
**Hapi** is a Python package providing fast and flexible way to build Hydrological models with different spatial representations (lumped, semidistributed and conceptual distributed) using HBV96.
The package is very flexible to an extent that it allows developers to change the structure of the defined conceptual model or to enter
their own model, it contains two routing functions muskingum cunge, and MAXBAS triangular function.




![1](/docs/img/Picture1.png)

![2](/docs/img/Picture2.png)

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


.. digraph:: Linking

    Hapi -> GIS;
    Hapi -> HBV;
    Hapi -> HBV;
    Hapi -> calibration;
    Hapi -> calibration
    Hapi -> distparameters
    Hapi -> distrrm
    Hapi -> giscatchment
    Hapi -> raster
    Hapi -> vector
    Hapi -> inputs
    Hapi -> performancecriteria
    Hapi -> routing
    Hapi -> run
    Hapi -> catchment
    Hapi -> statisticaltools
    Hapi -> wrapper
    Hapi -> weirdFn
    Hapi -> hbv_lake
    Hapi -> hbv_bergestrom92
    Hapi -> hminputs
    Hapi -> event
    Hapi -> river
    Hapi -> visualizer
    Hapi -> crosssection
    Hapi -> hmcalibration
    Hapi -> interface
    Hapi -> sensitivityanalysis
    Hapi -> remotesensing

Future work
-------------
  - Developing a regionalization method for connection model parameters with some catchment characteristics for better model calibration.
  - Developing and integrate river routing method (kinematic and diffusive wave approximation)
  - Apply the model for large scale (regional/continental) cases
  - Developing a DEM processing module for generating the river network at different DEM spatial resolutions.

References
==========

Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles, T. R. M. & Jaap Schellekens,  and L. A. B. (2016) Global-scale regionalization of hydrologic model parameters-Supporting materials 3599–3622. doi:10.1002/2015WR018247.Received

Bergström, S. (1992) The HBV model - its structure and applications. Smhi Rh 4(4), 35.

Rusli, S. R., Yudianto, D. & Liu, J. tao. (2015) Effects of temporal variability on HBV model calibration. Water Sci. Eng. 8(4), 291–300. Elsevier Ltd. doi:10.1016/j.wse.2015.12.002





.. toctree::
   :hidden:
   :maxdepth: 2

   installation.rst
   Inputs.rst
   Available Models <availablemodels.rst>
   tutorial.rst
