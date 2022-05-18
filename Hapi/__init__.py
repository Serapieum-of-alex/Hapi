"""
Created on Tue May 22 09:01:55 2018

@author: Mostafa
"""
try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# documentation format
__author__ = "Mostafa Farrag"
__email__ = 'moah.farag@gmail.com'
__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ()  # ("numpy", "pandas", "gdal")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))


# del hard_dependencies, dependency, missing_dependencies


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("gis")
    config.add_subpackage("hm")
    config.add_subpackage("plot")
    config.add_subpackage("rrm")
    config.add_subpackage("rs")
    config.add_subpackage("sm")


    return config


import Hapi.catchment
import Hapi.gis
import Hapi.plot
import Hapi.hm
import Hapi.java_functions

import Hapi.rrm
import Hapi.rs
import Hapi.run
import Hapi.sm
import Hapi.weirdFn

# import Hapi.saintvenant
# module level doc-string
__doc__ = """
Hapi - Hydrological library for Python
=====================================================================

**Hapi** is a Python package providing fast and flexible, way to build distributed
hydrological model using lumped conceptual model

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of rasters data downloaded from global data and easy way to
    manipulate the data to arrange it to run the model
  - Easy calibration of the model using Harmony search method and Genetic Algorithms
  - flexible GIS function to process rasters interpolate values and georeference
   calculated discharge values to the correct place

"""
# if __name__=='__main__':
#     print("Hapi")
