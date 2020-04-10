# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""

#%links
#from IPython import get_ipython   # to reset the variable explorer each time
#get_ipython().magic('reset -f')

#path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/05new_model/00inputs/"
path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/colombia/00inputs/GIS/4000/"

#%library
import numpy as np
import gdal


# functions
from Calibration import RunCalibration
#import Wrapper
#import GISpy as GIS
#import GISCatchment as GC
import DistParameters as DP
import PerformanceCriteria as PC

#%%
PrecPath = prec_path=path+"meteodata/4000/calib/prec"
Evap_Path = evap_path=path+"meteodata/4000/calib/evap"
TempPath = temp_path=path+"meteodata/4000/calib/temp"
DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = path+"GIS/4000/acc4000.tif"
FlowDPath = path+"GIS/4000/fd4000.tif"
Paths=[PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ]

#ParPathCalib = path+"meteodata/4000/"+"parameters.txt"
#ParPathRun = path+"meteodata/4000/parameters"
p2=[1, 227.31]
UB=[]
LB=[]
# define how generated parameters are going to be distributed spatially
# totaly distributed or totally distributed with some parameters are lumped
# for the whole catchment or HRUs or HRUs with some lumped parameters
SpatialVarFun=DP.par3d

soil_type=gdal.Open(path+"soil_classes.tif")
no_parameters=12
lumpedParNo=1
lumped_par_pos=[6]

SpatialVarArgs=[soil_type,no_parameters,lumpedParNo,lumped_par_pos]

# define the objective function and its arguments
objective_function=PC.RMSEHF
args=[1,1,0.75]
f=objective_function(np.array([1,2,3]),np.array([5,6,8]),*args)


OptimizationArgs=[]

# run calibration                
cal_parameters=RunCalibration(Paths,p2,Q_obs,UB,LB,SpatialVarFun,SpatialVarArgs,objective_function,printError=None,*args)
