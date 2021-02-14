"""
This code is used to Run the distributed model for jiboa rover in El Salvador
wher the catchment is consisted os a ustream lake and a volcanic area
-   you have to make the root directory to the examples folder to enable the code
    from reading input files

"""
from IPython import get_ipython
get_ipython().magic('reset -f')
import os
os.chdir("F:/02Case studies/El Salvador")

#%library
import gdal
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# HAPI modules
from Hapi.run import Run
import Hapi.hbv as HBV
import Hapi.performancecriteria as Pf
from Hapi.raster import Raster
#%%
"""
paths to meteorological data
"""
start = dt.datetime(2012,6,14,19,00,00)
end = dt.datetime(2014,11,17,00,00,00)
calib_end = dt.datetime(2013,12,23,00,00,00)

res = 4000

PrecPath = "inputs/Hapi/meteodata/"+str(res)+"/calib/prec_clipped"
Evap_Path = "inputs/Hapi/meteodata/"+str(res)+"/calib/evap_clipped"
TempPath = "inputs/Hapi/meteodata/"+str(res)+"/calib/temp_clipped"
#DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = "inputs/Hapi/GIS/"+str(res)+"_matched/acc"+str(res)+".tif"
FlowDPath = "inputs/Hapi/GIS/"+str(res)+"_matched/fd"+str(res)+".tif"
ParPath = "inputs/Hapi/meteodata/"+str(res)+"/parameters/"
#ParPath = "inputs/Hapi/meteodata/4000/"+"parameters.txt"
Paths=[PrecPath, Evap_Path, TempPath, FlowAccPath, FlowDPath, ]

#p2=[24, 1530]
#init_st=[0,5,5,5,0]
init_st = np.loadtxt("inputs/Hapi/meteodata/Initia-jiboa.txt", usecols=0).tolist()
snow = 0


# lake meteorological data
ind = pd.date_range(start, end, freq = "H" )
lakedata = pd.read_csv("inputs/Hapi/meteodata/lakedata.csv", index_col = 0)
lakedata.index = ind
lakeCalib = lakedata.loc[start:calib_end]
lakeValid = lakedata.loc[calib_end:end]
# convert the dataframe into array
lakeCalibArray = lakeCalib.values
# take only the plake, et, t and tm columns and exclude the last column
lakeCalibArray = lakeCalibArray[:,0:-1]

# where the lake discharges its flow (give the indices of the cell)
if res == 4000:
    lakecell = [2,1]    # 4km
elif res==2000:
    lakecell = [4,2]    # 2km
elif res == 1000:
    lakecell = [10,4]    # 1km
elif res == 500:
    lakecell = [19,10]    # 500m

LakeParameters = np.loadtxt("inputs/Hapi/meteodata/"+str(res)+"/Lakeparameters.txt").tolist()
StageDischargeCurve = np.loadtxt("inputs/Hapi/meteodata/curve.txt")
p2 = [1, 227.31, 133.98, 70.64]
Lake_init_st = np.loadtxt("inputs/Hapi/meteodata/Initia-lake.txt", usecols=0).tolist()

#%% run the model
Sim =pd.DataFrame(index = lakeCalib.index)
st, Sim['Q'], q_uz_routed, q_lz_trans = Run.RunHAPIwithLake(HBV, Paths, ParPath, p2, init_st,
                                                     snow, lakeCalibArray, StageDischargeCurve,
                                                     LakeParameters, lakecell,Lake_init_st)

#%% calculate some metrics
WS = {}
WS['type'] = 1
WS['N'] = 3
ModelMetrics=dict()
ModelMetrics['Calib_RMSEHF'] = round(Pf.RMSEHF(lakeCalib['Q'],Sim['Q'],WS['type'],WS['N'],0.75),3)
ModelMetrics['Calib_RMSELF'] = round(Pf.RMSELF(lakeCalib['Q'],Sim['Q'],WS['type'],WS['N'],0.75),3)
ModelMetrics['Calib_NSEHf'] = round(Pf.NSE(lakeCalib['Q'],Sim['Q']),3)
ModelMetrics['Calib_NSELf'] = round(Pf.NSE(np.log(lakeCalib['Q']),np.log(Sim['Q'])),3)
ModelMetrics['Calib_RMSE'] = round(Pf.RMSE(lakeCalib['Q'],Sim['Q']),3)
ModelMetrics['Calib_KGE'] = round(Pf.KGE(lakeCalib['Q'],Sim['Q']),3)
ModelMetrics['Calib_WB'] = round(Pf.WB(lakeCalib['Q'],Sim['Q']),3)

print(ModelMetrics)
#%% plotting
plt.figure(50,figsize=(15,8))
Sim.Q.plot(color=[(0,0.3,0.7)],linewidth=2.5,label="Observed data", zorder = 10)
ax1=lakeCalib['Q'].plot(color='#DC143C',linewidth=2.8,label='Simulated Calibration data')
ax1.annotate("Model performance" ,xy=('2012-12-01 00:00:00',20),fontsize=15)
ax1.annotate("RMSE = " + str(round(ModelMetrics['Calib_RMSE'],3)),xy=('2012-12-01 00:00:00',20-1.5),fontsize=15)
ax1.annotate("NSE = " + str(round(ModelMetrics['Calib_NSEHf'],2)),xy=('2012-12-01 00:00:00',20-3),fontsize=15)
plt.legend()
#ax1.annotate("RMSELF = " + str(round(committee['c_rmself'],3)),xy=('2013-01-01 00:00:00',max(calib['Q'])-3),fontsize=15)

#ax2=single_valid['Q'].plot(color='orange',linewidth=2.8,label='Simulated Validation')
#ax2.annotate("Model performance" ,xy=('2014-01-01 00:00:00',20),fontsize=15)
#ax2.annotate("RMSE = " +str(round(single['v_rmse'],3)),xy=('2014-01-01 00:00:00',20-1.5),fontsize=15)
#ax1.annotate("NSE = " + str(round(single['v_nsehf'],2)),xy=('2014-01-01 00:00:00',20-3),fontsize=15)
#ax2.annotate("RMSELF = " +str(round(committee['v_rmself'],3)),xy=('2014-12-01 00:00:00',max(calib['Q'])-3),fontsize=15)
#%% store the result into rasters
# create list of names
src=gdal.Open(FlowAccPath)

index=pd.date_range(start,calib_end,freq="1H")

resultspath="results/upper_zone_discharge/4000/"
names=[resultspath+str(i)[:-6] for i in index]
names=[i.replace("-","_") for i in names]
names=[i.replace(" ","_") for i in names]
names=[i+".tif" for i in names]

"""
to save the upper zone discharge distributerd discharge in a raster forms
uncomment the next line
"""
Raster.RastersLike(src,q_uz_routed[:,:,:-1],names)
