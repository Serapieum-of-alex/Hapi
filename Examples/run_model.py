"""
This code is used to calibrate the model

-   you have to make the root directory to the examples folder to enable the code
    from reading input files

"""
#%library
import gdal
import datetime as dt
import pandas as pd
#import matplotlib.pyplot as plt

# HAPI modules
from Hapi.run import RunModel
import Hapi.hbv as HBV
import Hapi.raster as GIS
#%%
"""
paths to meteorological data
"""
PrecPath = prec_path="data/meteodata/4000/calib/prec"
Evap_Path = evap_path="data/meteodata/4000/calib/evap"
TempPath = temp_path="data/meteodata/4000/calib/temp"
#DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = "data/GIS/4000/acc4000.tif"
FlowDPath = "data/GIS/4000/fd4000.tif"
ParPathCalib = "data/meteodata/4000/"+"parameters.txt"
ParPathRun = "data/results/parameters/4000"
Paths=[PrecPath, Evap_Path, TempPath, FlowAccPath, FlowDPath, ]

#p2=[1, 227.31]
p2=[24, 1530]
init_st=[0,5,5,5,0]
snow = 0

st, q_out, q_uz_routed, q_lz_trans = RunModel(HBV,Paths,ParPathRun,p2,init_st,snow)

#%% store the result into rasters
# create list of names
src=gdal.Open(FlowAccPath)
s=dt.datetime(2009,01,1,00,00,00)
e=dt.datetime(2011,12,31,00,00,00)
index=pd.date_range(s,e,freq="1D")
resultspath="data/results/upper_zone_discharge/4000/"
names=[resultspath+str(i)[:-6] for i in index]
names=[i.replace("-","_") for i in names]
names=[i.replace(" ","_") for i in names]
names=[i+".tif" for i in names]

"""
to save the upper zone discharge distributerd discharge in a raster forms
uncomment the next line
"""
#GIS.RastersLike(src,q_uz_routed[:,:,:-1],names)
