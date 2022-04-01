""" Distributed model with a maxbas routing scheme """
import os
# comp = "F:/02Case studies/"
Comp = r"C:\MyComputer\01Algorithms\Hydrology\Hapi/"
os.chdir(Comp+ "examples/")
# import numpy as np
from osgeo import gdal
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from Hapi.run import Run
from Hapi.catchment import Catchment
import Hapi.rrm.hbv_bergestrom92 as HBV
import Hapi.sm.performancecriteria as PC
#%% Paths
path = Comp + "/Coello/Hapi/Data/00inputs/"
PrecPath = path + "meteodata/4000/calib/prec-MSWEP"
Evap_Path = path + "meteodata/4000/calib/evap"
TempPath = path + "meteodata/4000/calib/temp"
FlowAccPath = path + "GIS/4000/acc4000.tif"
# FlowDPath = path+"GIS/4000/fd4000.tif"
# FlowPathLengthPath = path + "GIS/4000/FPL4000.tif"
# ParPath = "results/parameters/4000/lumped/2021-03-09/rasters"
ParPath = "F:/Users/mofarrag/coello/Hapi/Data/00inputs/Basic_inputs/default parameters/initial"
#%% Meteorological data
AreaCoeff = 1530
InitialCond = [0,5,5,5,0]
Snow = 0
"""
Create the model object and read the input data
"""
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end, SpatialResolution = "Distributed")
Coello.ReadRainfall(PrecPath)
Coello.ReadTemperature(TempPath)
Coello.ReadET(Evap_Path)

Coello.ReadFlowAcc(FlowAccPath)
# Coello.ReadFlowDir(FlowDPath)
# Coello.ReadFlowPathLength(FlowPathLengthPath)

Coello.ReadParameters(ParPath, Snow,Maxbas=True)
Coello.ReadLumpedModel(HBV, AreaCoeff, InitialCond)
#%% Gauges
Coello.ReadGaugeTable(path+"Discharge/stations/gauges.csv", FlowAccPath)
GaugesPath = path+"Discharge/stations/"
Coello.ReadDischargeGauges(GaugesPath, column='id', fmt="%Y-%m-%d")
#%% Run the model
"""
Outputs:
    ----------
    1-statevariables: [numpy attribute] 
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute] 
        3D array of the lower zone discharge
    3-quz: [numpy attribute] 
        3D array of the upper zone discharge
    4-qout: [numpy attribute] 
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute] 
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute] 
        3D array of the lower zone discharge translated at each time step
"""
Run.RunFW1(Coello)
#%% calculate performance criteria
Coello.ExtractDischarge(CalculateMetrics=True, FW1=True)

gaugeid = Coello.GaugesTable.loc[Coello.GaugesTable.index[-1],'id']
print("----------------------------------")
print("Gauge - " +str(gaugeid))
print("RMSE= " + str(round(Coello.Metrics.loc['RMSE',gaugeid],2)))
print("NSE= " + str(round(Coello.Metrics.loc['NSE',gaugeid],2)))
print("NSEhf= " + str(round(Coello.Metrics.loc['NSEhf',gaugeid],2)))
print("KGE= " + str(round(Coello.Metrics.loc['KGE',gaugeid],2)))
print("WB= " + str(round(Coello.Metrics.loc['WB',gaugeid],2)))    
#%% plot
i = 5
gaugei = 5
plotstart = "2009-01-01"
plotend = "2011-12-31"

Coello.PlotHydrograph(plotstart, plotend, gaugei)
#%% store the result into rasters
# create list of names
src=gdal.Open(FlowAccPath)
s=dt.datetime(2012,6,14,19,00,00)
e=dt.datetime(2013,12,23,00,00,00)
index=pd.date_range(s,e,freq="1H")
resultspath="results/"
names=[resultspath+str(i)[:-6] for i in index]
names=[i.replace("-","_") for i in names]
names=[i.replace(" ","_") for i in names]
names=[i+".tif" for i in names]

#Raster.RastersLike(src,q_uz_routed[:,:,:-1],names)
