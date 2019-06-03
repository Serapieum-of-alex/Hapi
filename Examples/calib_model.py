# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
#%links
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
#import os
#os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Model/Code/colombia")
#import sys
#sys.path.append("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function")
#path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/05new_model/00inputs/"
#path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/colombia/00inputs/" #GIS/4000/

#%library
import numpy as np
import pandas as pd
from datetime import datetime
import gdal
#from pyOpt import Optimization, ALHSO,Optimizer

# functions
from Hapi.Calibration import RunCalibration
import Hapi.HBV as HBV
#import Wrapper
#import Hapi.GISpy as GIS
import Hapi.GISCatchment as GC
import Hapi.DistParameters as DP
import Hapi.PerformanceCriteria as PC
#import Inputs
#%%

### Meteorological & GIS Data 
# resolution of input data is 4km*4km
PrecPath = prec_path="data/meteodata/calib/prec"
Evap_Path = evap_path="data/meteodata/calib/evap"
TempPath = temp_path="data/meteodata/calib/temp"
#DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = "data/GIS/acc4000.tif"
FlowDPath = "data/GIS/fd4000.tif"
Paths=[PrecPath, Evap_Path, TempPath, FlowAccPath, FlowDPath, ]

#ParPathCalib = path+"meteodata/4000/"+"parameters.txt"
#ParPathRun = path+"meteodata/4000/parameters"

###  Boundaries, p2
p2=[24, 1530]
#[sp,sm,uz,lz,wc]
init_st=[0,5,5,5,0]
snow=0
UB=np.loadtxt("data/UB.txt", usecols=0)
LB=np.loadtxt("data/LB.txt", usecols=0)

Basic_inputs=dict(p2=p2, init_st=init_st, UB=UB, LB=LB, snow=snow)


### spatial variability function
""" 
define how generated parameters are going to be distributed spatially
totaly distributed or totally distributed with some parameters are lumped
for the whole catchment or HRUs or HRUs with some lumped parameters 
for muskingum parameters k & x include the upper and lower bound in both 
UB & LB with the order of Klb then kub 
function inside the calibration algorithm is written as following
par_dist=SpatialVarFun(par,*SpatialVarArgs,kub=kub,klb=klb)    

"""
SpatialVarFun=DP.par3dLumped
raster=gdal.Open(FlowAccPath)
no_parameters=12
SpatialVarArgs=[raster,no_parameters]

### Objective function
# stations discharge
Sdate='2009-01-01'
Edate='2011-12-31'
Qobs = pd.read_csv("data/Discharge/Headflow.txt",header=0 ,delimiter="\t", skiprows=11, 
                   engine='python',index_col=0)
ind=[datetime(int(i.split("/")[0]),int(i.split("/")[1]),int(i.split("/")[2]))  for i in Qobs.index.tolist()]
Qobs.index=ind
Qobs =Qobs.loc[Sdate:Edate]

# outlet discharge    
Qobs[6] =np.loadtxt("data/Discharge/Qout_c.txt")
Qobs=Qobs.as_matrix()

stations=pd.read_excel("data/Discharge/stations/4000/Q.xlsx",sheetname="coordinates",convert_float=True)
coordinates=stations[['id','x','y','weight']][:]

# calculate the nearest cell to each station
coordinates.loc[:,["cell_row","cell_col"]]=GC.NearestCell(raster,coordinates)

acc=gdal.Open(FlowAccPath ) 
acc_A=acc.ReadAsArray()
# define the objective function and its arguments
OF_args=[coordinates]

"""
OF is the objective function used for the calibration 


"""
def OF(Qobs,Qout,q_uz_routed,q_lz_trans,coordinates):
    all_errors=[]
    # error for all internal stations
    for i in range(len(coordinates)-1):
        Quz=np.reshape(q_uz_routed[int(coordinates.loc[coordinates.index[i],"cell_row"]),int(coordinates.loc[coordinates.index[i],"cell_col"]),:-1],len(Qobs))
        Qlz=np.reshape(q_lz_trans[int(coordinates.loc[coordinates.index[i],"cell_row"]),int(coordinates.loc[coordinates.index[i],"cell_col"]),:-1],len(Qobs))
        Q=Quz+Qlz
        all_errors.append((PC.RMSE(Qobs[:,i],Q))*coordinates.loc[coordinates.index[i],'weight'])
    #outlet observed discharge is at the end of the array
    all_errors.append((PC.RMSE(Qobs[:,-1],Qout))*coordinates.loc[coordinates.index[-1],'weight'])
    print(all_errors)
    error=sum(all_errors)
    return error

### Optimization
store_history=1
history_fname="par_history.txt"
OptimizationArgs=[store_history,history_fname]
#%%
# run calibration                
cal_parameters=RunCalibration(HBV, Paths, Basic_inputs,
                              SpatialVarFun, SpatialVarArgs, 
                              OF,OF_args,Qobs, 
                              OptimizationArgs,
                              printError=1)
#%% convert parameters to rasters
ParPath = "par15_7_2018.txt"
par=np.loadtxt(ParPath)
klb=0.5
kub=1
Path="parameters/"

#DP.SaveParameters(SpatialVarFun, raster, par, no_parameters,snow ,kub, klb,Path)
