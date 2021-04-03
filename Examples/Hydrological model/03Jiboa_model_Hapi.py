"""
This code is used to Run the distributed model for jiboa rover in El Salvador
wher the catchment is consisted os a ustream lake and a volcanic area
-   you have to make the root directory to the examples folder to enable the code
    from reading input files

"""
# from IPython import get_ipython
# get_ipython().magic('reset -f')
import os
os.chdir("F:/02Case studies/El Salvador")

#%library
import gdal
# import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# HAPI modules
from Hapi.run import Run, Model, Lake
import Hapi.hbv as HBV
import Hapi.hbv_lake as HBVLake
import Hapi.performancecriteria as Pf
from Hapi.raster import Raster
#%% Paths
res = 4000
"""
paths to meteorological data
"""
PrecPath = "inputs/Hapi/meteodata/"+str(res)+"/calib/prec_clipped"
Evap_Path = "inputs/Hapi/meteodata/"+str(res)+"/calib/evap_clipped"
TempPath = "inputs/Hapi/meteodata/"+str(res)+"/calib/temp_clipped"
FlowAccPath = "inputs/Hapi/GIS/"+str(res)+"_matched/acc"+str(res)+".tif"
FlowDPath = "inputs/Hapi/GIS/"+str(res)+"_matched/fd"+str(res)+".tif"
ParPath = "inputs/Hapi/meteodata/"+str(res)+"/parameters/"
# Lake
LakeMeteoPath = "inputs/Hapi/meteodata/lakedata.csv"
LakeParametersPath = "inputs/Hapi/meteodata/"+str(res)+"/Lakeparameters.txt"
GaugesPath = "inputs/Hapi/meteodata/Gauges/"#Station.csv
#%% Distributed Model Object

AreaCoeff = 227.31
InitialCond = np.loadtxt("inputs/Hapi/meteodata/Initia-jiboa.txt", usecols=0).tolist()
Snow = 0

Sdate = '2012-06-14 19:00:00'
# Edate = '2014-11-17 00:00:00'
Edate = '2013-12-23 00:00:00'
name = "Jiboa"
Jiboa = Model(name, Sdate, Edate, SpatialResolution = "Distributed",
              TemporalResolution = "Hourly", fmt='%Y-%m-%d %H:%M:%S')
Jiboa.ReadRainfall(PrecPath)
Jiboa.ReadTemperature(TempPath)
Jiboa.ReadET(Evap_Path)
Jiboa.ReadFlowAcc(FlowAccPath)
Jiboa.ReadFlowDir(FlowDPath)
Jiboa.ReadParameters(ParPath)
Jiboa.ReadLumpedModel(HBV, AreaCoeff, InitialCond, Snow)
#%% Lake Object
"""
lake meteorological data
"""
# where the lake discharges its flow (give the indices of the cell)
if res == 4000:
    OutflowCell = [2,1]    # 4km
elif res==2000:
    OutflowCell = [4,2]    # 2km
elif res == 1000:
    OutflowCell = [10,4]    # 1km
elif res == 500:
    OutflowCell = [19,10]    # 500m

Sdate = '2012.06.14 19:00:00'
# Edate = '2014.11.17 00:00:00'
Edate = '2013.12.23 00:00:00'

JiboaLake = Lake(StartDate=Sdate, EndDate=Edate, fmt='%Y.%m.%d %H:%M:%S',
                 TemporalResolution="Hourly", Split=True)

JiboaLake.ReadMeteoData(LakeMeteoPath,fmt='%d.%m.%Y %H:%M')
JiboaLake.ReadParameters(LakeParametersPath)


StageDischargeCurve = np.loadtxt("inputs/Hapi/meteodata/curve.txt")
LakeInitCond = np.loadtxt("inputs/Hapi/meteodata/Initia-lake.txt", usecols=0).tolist()
LakeCatArea = 133.98
LakeArea = 70.64
Snow = 0
JiboaLake.ReadLumpedModel(HBVLake, LakeCatArea, LakeArea, LakeInitCond,
                          OutflowCell, StageDischargeCurve, Snow)
#%% Gauges
Date1 = '14.06.2012 19:00'
Date2 = '23.12.2013 00:00'
Jiboa.ReadGaugeTable(GaugesPath+"GaugesTable.csv",FlowAccPath)
Jiboa.ReadDischargeGauges(GaugesPath, column='id', fmt='%d.%m.%Y %H:%M',
                          Split=True, Date1=Date1, Date2=Date2)
#%% run the model
# Sim =pd.DataFrame(index = JiboaLake.Index)
Run.RunHAPIwithLake(Jiboa, JiboaLake)
#%% calculate some metrics
Qobs = Jiboa.QGauges[Jiboa.GaugesTable.loc[0,'id']]

WS = {}
WS['type'] = 1
WS['N'] = 3
ModelMetrics=dict()
ModelMetrics['Calib_RMSEHF'] = round(Pf.RMSEHF(Qobs,Sim['Q'],WS['type'],WS['N'],0.75),3)
ModelMetrics['Calib_RMSELF'] = round(Pf.RMSELF(Qobs,Sim['Q'],WS['type'],WS['N'],0.75),3)
ModelMetrics['Calib_NSEHf'] = round(Pf.NSE(Qobs,Sim['Q']),3)
ModelMetrics['Calib_NSELf'] = round(Pf.NSE(np.log(Qobs),np.log(Sim['Q'])),3)
ModelMetrics['Calib_RMSE'] = round(Pf.RMSE(Qobs,Sim['Q']),3)
ModelMetrics['Calib_KGE'] = round(Pf.KGE(Qobs,Sim['Q']),3)
ModelMetrics['Calib_WB'] = round(Pf.WB(Qobs,Sim['Q']),3)

print(ModelMetrics)
#%% plotting
plt.figure(50,figsize=(15,8))
Sim.Q.plot(color=[(0,0.3,0.7)],linewidth=2.5,label="Observed data", zorder = 10)
ax1=Qobs.plot(color='#DC143C',linewidth=2.8,label='Simulated Calibration data')
ax1.annotate("Model performance" ,xy=('2012-12-01 00:00:00',20),fontsize=15)
ax1.annotate("RMSE = " + str(round(ModelMetrics['Calib_RMSE'],3)),xy=('2012-12-01 00:00:00',20-1.5),fontsize=15)
ax1.annotate("NSE = " + str(round(ModelMetrics['Calib_NSEHf'],2)),xy=('2012-12-01 00:00:00',20-3),fontsize=15)
ax1.set_xlabel("Date", fontsize='15')
ax1.set_ylabel("Discharge m3/s", fontsize='15')
plt.tight_layout()
plt.legend()
# ax1.annotate("RMSELF = " + str(round(committee['c_rmself'],3)),xy=('2013-01-01 00:00:00',max(calib['Q'])-3),fontsize=15)

#ax2=single_valid['Q'].plot(color='orange',linewidth=2.8,label='Simulated Validation')
#ax2.annotate("Model performance" ,xy=('2014-01-01 00:00:00',20),fontsize=15)
#ax2.annotate("RMSE = " +str(round(single['v_rmse'],3)),xy=('2014-01-01 00:00:00',20-1.5),fontsize=15)
#ax1.annotate("NSE = " + str(round(single['v_nsehf'],2)),xy=('2014-01-01 00:00:00',20-3),fontsize=15)
#ax2.annotate("RMSELF = " +str(round(committee['v_rmself'],3)),xy=('2014-12-01 00:00:00',max(calib['Q'])-3),fontsize=15)
#%% store the result into rasters
# create list of names
src=gdal.Open(FlowAccPath)

# index=pd.date_range(Jiboa.StartDate,Jiboa.EndDate,freq="1H")

resultspath="results/upper_zone_discharge/4000/"
names=[resultspath+str(i)[:-6] for i in Jiboa.Index]
names=[i.replace("-","_") for i in names]
names=[i.replace(" ","_") for i in names]
names=[i+".tif" for i in names]

"""
to save the upper zone discharge distributerd discharge in a raster forms
uncomment the next line
"""
Raster.RastersLike(src,q_uz_routed[:,:,:-1],names)
