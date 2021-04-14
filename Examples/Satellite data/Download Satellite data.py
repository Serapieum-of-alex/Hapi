"""
Download Satellite data
ECMWF
Installation of ECMWF API key
1 - to be able to use Hapi to download ECMWF data you need to register and setup your account in the ECMWF website (https://apps.ecmwf.int/registration/)

2 - Install ECMWF key (instruction are here https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key)

Using ResmoteSensing module from Hapi

"""

from Hapi.remotesensing import RemoteSensing as RS
from Hapi.remotesensing import Variables, CHIRPS

wpath = "/data/Satellite data/"
#%% precipitation
# # chrips
StartDate = '2009-01-01'
EndDate = '2009-02-01'
Time = 'daily'
latlim = [4.190755,4.643963]
lonlim = [-75.649243,-74.727286]
Path = wpath + "trial/"
#%%
Vars = Variables('daily')
Vars.__str__()
#%% Temperature
StartDate = '2009-01-01'
EndDate = '2009-02-01'
Time = 'daily'
latlim = [4.190755,4.643963]
lonlim = [-75.649243,-74.727286]
Path = wpath + "trial/"
# Temperature, Evapotranspiration
variables = ['T','E']

Coello = RS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim , lonlim=lonlim, Path=Path, Vars=variables)

Coello.ECMWF()
#%%
Coello = CHIRPS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim , lonlim=lonlim, Path=Path)
Coello.Download() #cores=4
