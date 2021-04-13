"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""
# Libraries
from Hapi.remotesensing import RemoteSensing as RS
from Hapi.remotesensing import Variables
from Hapi.remotesensing import CHIRPS
#%% Basin data
StartDate = '2009-01-01'
EndDate = '2009-02-01'
Time = 'daily'
latlim = [4.190755,4.643963]
lonlim = [-75.649243,-74.727286]
# make sure to provide a full path not relative path
# please replace the following root_path to the repo main directory in your machine
root_path = "F:/01Algorithms/Hydrology/HAPI/"
Path = root_path + "Examples/data/satellite_data/"
#%%
"""
check the ECMWF variable names that you have to provide to the RemoteSensing object
"""
Vars = Variables('daily')
Vars.__str__()
#%% ECMWF
"""
provide the time period, temporal resolution, extent and variables of your interest
"""
StartDate = '2009-01-01'
EndDate = '2009-00-10'
Time = 'daily'
latlim = [4.190755,4.643963]
lonlim = [-75.649243,-74.727286]
# Temperature, Evapotranspiration
variables = ['T','E']

Coello = RS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim , lonlim=lonlim, Path=Path, Vars=variables)

# Coello.ECMWF()
#%% CHRIPS
Coello = CHIRPS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim , lonlim=lonlim, Path=Path)
Coello.Download()
#%%
"""
if you want to use parallel downloads using multi cores, enter the number of
cores you want to use

PS. the multi-coredownload does not have an indication bar
"""
cores = 4

Coello = CHIRPS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim, lonlim=lonlim, Path=Path)
Coello.Download(cores=cores)