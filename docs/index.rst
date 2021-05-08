  .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4686056.svg
   :target: https://doi.org/10.5281/zenodo.4686056

Current release info
====================

  .. image:: https://img.shields.io/pypi/pyversions/hapi-nile   :alt: PyPI - Python Version

  .. image:: https://img.shields.io/github/commit-activity/m/mafarrag/HAPI   :alt: GitHub commit activity
  
  .. image:: https://img.shields.io/github/issues/mafarrag/HAPI :alt: GitHub issues




	.. image:: https://anaconda.org/conda-forge/hapi/badges/downloads.svg :target: https://anaconda.org/conda-forge/hapi

  .. image:: https://img.shields.io/conda/vn/conda-forge/hapi   :alt: Conda (channel only)     

  .. image:: https://img.shields.io/gitter/room/mafarrag/Hapi   :alt: Gitter

  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads

   .. image:: https://img.shields.io/pypi/v/hapi-nile   :alt: PyPI |  
   
   .. image:: https://anaconda.org/conda-forge/hapi/badges/platforms.svg   :target: https://anaconda.org/conda-forge/hapi


  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 


  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 

  .. image:: https://static.pepy.tech/personalized-badge/hapi-nile?period=week&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
 



  
  .. image:: /img/Hapi.png
    :width: 400pt
  
 
  .. image:: /img/name.png
    :width: 400pt


Hapi - Hydrological library for Python 
=====================================================================
**Hapi** is a Python package providing fast and flexible way to build Hydrological models with different spatial representations (lumped, semidistributed and conceptual distributed) using HBV96.
The package is very flexible to an extent that it allows developers to change the structure of the defined conceptual model or to enter
their own model, it contains two routing functions muskingum cunge, and MAXBAS triangular function.



  .. image:: /img/Picture1.png
   :width: 400pt

  .. image:: /img/Picture2.png
   :width: 400pt

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
    Hapi -> HM;
    Hapi -> RemoteSensing;
    Hapi -> RRM;
    Hapi -> Statistics;
    Hapi -> catchment;
    Hapi -> weirdFn;
    Hapi -> visualizer;
    GIS -> raster;
    GIS -> vector;
    GIS -> giscatchment;
    RRM -> HBV;
    RRM -> calibration;
    RRM -> distparameters;
    RRM -> distrrm;
    RRM -> routing;
    RRM -> run;
    RRM -> inputs;
    RRM -> wrapper;
    RRM -> hbv_lake;
    RRM -> hbv_bergestrom92;
    HM -> inputs;
    HM -> event;
    HM -> river;
    HM -> calibration;
    HM -> crosssection;
    HM -> interface;
    Statistics -> performancecriteria;
    Statistics -> statisticaltools;
    Statistics -> sensitivityanalysis;
    RemoteSensing -> remotesensing;
    dpi=69;

Future work
-------------
  - Developing a regionalization method for connection model parameters with some catchment characteristics for better model calibration.
  - Developing and integrate river routing method (kinematic and diffusive wave approximation)
  - Apply the model for large scale (regional/continental) cases
  - Developing a DEM processing module for generating the river network at different DEM spatial resolutions.

References
==========

Farrag, M. & Corzo, G. (2021) MAfarrag/Hapi: Hapi. doi:10.5281/ZENODO.4662170

Farrag, M., Perez, G. C. & Solomatine, D. (2021) Spatio-Temporal Hydrological Model Structure and Parametrization Analysis. J. Mar. Sci. Eng. 9(5), 467. doi:10.3390/jmse9050467

Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles, T. R. M. & Jaap Schellekens,  and L. A. B. (2016) Global-scale regionalization of hydrologic model parameters-Supporting materials 3599–3622. doi:10.1002/2015WR018247.Received

Bergström, S. (1992) The HBV model - its structure and applications. Smhi Rh 4(4), 35.

Rusli, S. R., Yudianto, D. & Liu, J. tao. (2015) Effects of temporal variability on HBV model calibration. Water Sci. Eng. 8(4), 291–300. Elsevier Ltd. doi:10.1016/j.wse.2015.12.002

