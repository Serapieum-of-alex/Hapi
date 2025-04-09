"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/example

install and use earth2observe package https://github.com/MAfarrag/earth2observe
"""

from earth2observe.chirps import CHIRPS, Catalog
from earth2observe.earth2observe import Earth2Observe
from earth2observe.ecmwf import Catalog

root_path = "C:/MyComputer/01Algorithms/Hydrology/Hapi/"
# %% Basin data
start = "2009-01-01"
end = "2009-02-01"
temporal_resolution = "daily"
latlim = [4.190755, 4.643963]
lonlim = [-75.649243, -74.727286]

# make sure to provide a full path not relative path
# please replace the following root_path to the repo main directory in your machine
path = root_path + "examples/data/satellite_data/"
# %%
"""
check the ECMWF variable names that you have to provide to the RemoteSensing object
"""
var = "T"
catalog = Catalog()
print(catalog.catalog)
catalog.get_variable(var)
# %% pripitation data from ecmwf
"""
provide the time period, temporal resolution, extent and variables of your interest
"""
# Temperature, Evapotranspiration
variables = ["T", "E"]
source = "ecmwf"
e2o = Earth2Observe(
    data_source=source,
    start=start,
    end=end,
    variables=variables,
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path,
)

e2o.download()
# %% CHRIPS
variables = ["precipitation"]
e2o = Earth2Observe(
    data_source=source,
    start=start,
    end=end,
    variables=variables,
    lat_lim=latlim,
    lon_lim=lonlim,
    temporal_resolution=temporal_resolution,
    path=path,
)
e2o.download()
# %%
"""
if you want to use parallel downloads using multi cores, enter the number of
cores you want to use

PS. the multi-coredownload does not have an indication bar
"""
e2o.download(cores=4)
