# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
#%links
#from IPython import get_ipython   # to reset the variable explorer each time
#get_ipython().magic('reset -f')
import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Model/colombia")
import sys
sys.path.append("C:/Users/Mostafa/Desktop/My Files/Research/Hapi")
path="C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Data/colombia/00inputs/"
#%library
import numpy as np
import pandas as pd
# functions
import HBVLumped
from Calibration import LumpedCalibration
from Routing import TriangularRouting 
import Hapi.PerformanceCriteria as PC
#%%
### meteorological data
path="C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Data/colombia/00inputs/Lumped/"
data=pd.read_csv(path+"meteo_data.txt",header=0 ,delimiter=',',#"\t", #skiprows=11, 
                   engine='python',index_col=0)
data=data.as_matrix()

### Basic_inputs
ConceptualModel=HBVLumped

p2=[24, 1530]
init_st=[0,5,5,5,0]
snow = 0

UB=np.loadtxt("Basic_inputs/UB_Lumped.txt", usecols=0)
LB=np.loadtxt("Basic_inputs/LB_Lumped.txt", usecols=0)

# Routing
Routing=1
RoutingFn=TriangularRouting

Basic_inputs=dict(p2=p2, init_st=init_st, UB=UB, LB=LB, snow=snow, 
                  Routing=Routing, RoutingFn=RoutingFn)

### Objective function
# outlet discharge
#Sdate='2009-01-01'
#Edate='2011-12-31'
Qobs =np.loadtxt(path+"Qout_c.txt")

# define the objective function and its arguments
OF_args=[]
OF=PC.RMSE

### Optimization
store_history=1
history_fname="par_history.txt"
OptimizationArgs=[store_history,history_fname]
#%%
# run calibration                
cal_parameters=LumpedCalibration(ConceptualModel, data, Basic_inputs,
                   OF, OF_args, Qobs, OptimizationArgs, printError=None)
#%% convert parameters to rasters
ParPath = "par15_7_2018.txt"
par=np.loadtxt(ParPath)
klb=0.5
kub=1
Path="parameters/"

#DP.SaveParameters(SpatialVarFun, raster, par, no_parameters,snow ,kub, klb,Path)
