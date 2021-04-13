# import watools as wa
from Hapi.remotesensing import RemoteSensing as RS, Variables
from Hapi.remotesensing import CHIRPS

# wpath = "F:/02Case studies/Coello/Hapi/Data/"
wpath = "F:/02Case studies/Coello/Hapi/Data/00/"
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
# StartDate = '2009-01-01'
# EndDate = '2009-02-01'
# Time = 'daily'
# latlim = [4.190755,4.643963]
# lonlim = [-75.649243,-74.727286]
# Path = wpath + "trial/"
# # Temperature, Evapotranspiration
# variables = ['T','E']

# Coello = RS(StartDate=StartDate, EndDate=EndDate, Time=Time,
#             latlim=latlim , lonlim=lonlim, Path=Path, Vars=variables)

# Coello.ECMWF()
#%%
Coello = CHIRPS(StartDate=StartDate, EndDate=EndDate, Time=Time,
            latlim=latlim , lonlim=lonlim, Path=Path)
Coello.Download() #cores=4
#%% Soil Type
# RS.main(Dir=wpath,Vars=['SLT'],
#            latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
#            Startdate='2009-01-01', Enddate='2013-12-31')
#%% evapotranspiration
# ETref=wa.Products.ETref
# Ett=ETref.daily(Dir=wpath,
#                 latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
#                 Startdate='2009-01-01', Enddate='2013-12-31')
