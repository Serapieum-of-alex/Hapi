# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
#%links
#from IPython import get_ipython   # to reset the variable explorer each time
#get_ipython().magic('reset -f')
import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Model/Code/colombia")
import sys
sys.path.append("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function")
#path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/05new_model/00inputs/"
path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/colombia/00inputs/" #GIS/4000/
#%library
import numpy as np
import pandas as pd
from datetime import datetime
import gdal
#from pyOpt import Optimization, ALHSO,Optimizer

# functions
from Calibration import RunCalibration
#import Wrapper
import GISpy as GIS
import GISCatchment as GC
import DistParameters as DP
import PerformanceCriteria as PC

#%%
### Meteorological & GIS Data 
PrecPath = prec_path=path+"meteodata/4000/prec"
Evap_Path = evap_path=path+"meteodata/4000/evap"
TempPath = temp_path=path+"meteodata/4000/temp"
DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = path+"GIS/4000/acc4000.tif"
FlowDPath = path+"GIS/4000/fd4000.tif"
Paths=[PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ]

#ParPathCalib = path+"meteodata/4000/"+"parameters.txt"
#ParPathRun = path+"meteodata/4000/parameters"

###  Boundaries, p2
p2=[1, 227.31]
Qobs=[]
UB=np.loadtxt("UB.txt", usecols=0)
LB=np.loadtxt("LB.txt", usecols=0)

### spatial variability function
""" 
define how generated parameters are going to be distributed spatially
totaly distributed or totally distributed with some parameters are lumped
for the whole catchment or HRUs or HRUs with some lumped parameters 
for muskingum parameters k & x include the upper and lower bound in both 
UB & LB with the order of Klb then kub 
"""
SpatialVarFun=DP.par3dLumped
soil_type=gdal.Open(path+"GIS/4000/soil_classes.tif")
no_parameters=12
SpatialVarArgs=[soil_type,no_parameters]
#par_g=np.random.random(no_parameters) #no_elem*(no_parameters-no_lumped_par)
#klb=0.5
#kub=1
#f=SpatialVarFun(par_g,*SpatialVarArgs,kub=kub,klb=klb)

#lumpedParNo=1
#lumped_par_pos=[6]
#SpatialVarArgs=[soil_type,no_parameters,lumpedParNo,lumped_par_pos]


### Objective function
Qobs = pd.read_csv(path+"Discharge/Headflow.txt",header=0 ,delimiter="\t", skiprows=11, 
                   engine='python',index_col=0)
ind=[datetime(int(i.split("/")[0]),int(i.split("/")[1]),int(i.split("/")[2]))  for i in Qobs.index.tolist()]
Qobs.index=ind

stations=pd.read_excel(path+"Discharge/Q.xlsx",sheetname="coordinates",convert_float=True)
coordinates=stations[['id','x','y']][:]
raster=soil_type
# calculate the nearest cell to each station
coordinates.loc[:,["cell_row","cell_col"]]=GC.NearestCell(raster,coordinates)

# define the objective function and its arguments
#objective_function=PC.RMSEHF
def objective_function():

    pp_idw=np.reshape(prec_ISDW[:,x,y],len(prec_data))#.tolist()#[0]
    ee_idw=np.reshape(et_ISDW[:,x,y],len(prec_data))#.tolist()[0]

    
    



args=[1,1,0.75]
#f=objective_function(np.array([1,2,3]),np.array([5,6,8]),*args)



OptimizationArgs=[]

# run calibration                
cal_parameters=RunCalibration(Paths,p2,Qobs,UB,LB,SpatialVarFun,SpatialVarArgs,objective_function,printError=None,*args)

#%% convert parameters to rasters
#DemPath = path+"GIS/4000/dem4000.tif"
#dem=gdal.Open(DemPath)
#ParPath = path+"meteodata/4000/"+"parameters.txt"
#par=np.loadtxt(ParPath)
#klb=par[-2]
#kub=par[-1]
#par=par[:-2]
#
#par2d=Df.par2d_lumpedK1(par,dem,12,kub,klb)
## save 
#pnme=["FC", "BETA", "ETF", "LP", "CFLUX", "K", "K1","ALPHA", "PERC", "Pcorr", "Kmuskingum", "Xmuskingum"]
#for i in range(np.shape(par2d)[2]):
#    GIS.RasterLike(dem,par2d[:,:,i],"parameters/"+pnme[i]+".tif")
