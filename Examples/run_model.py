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
path="C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Data/colombia/00inputs/"

#%library
import numpy as np
import gdal
import datetime as dt
import pandas as pd
#import matplotlib.pyplot as plt

# functions
#import Wrapper
#import DistParameters as Df
import GISpy as GIS
#import GISCatchment as GC
from RUN import RunModel
import HBV
#%%
PrecPath = prec_path=path+"meteodata/4000/calib/prec"
Evap_Path = evap_path=path+"meteodata/4000/calib/evap"
TempPath = temp_path=path+"meteodata/4000/calib/temp"
#DemPath = path+"GIS/4000/dem4000.tif"
FlowAccPath = path+"GIS/4000/acc4000.tif"
FlowDPath = path+"GIS/4000/fd4000.tif"
ParPathCalib = path+"meteodata/4000/"+"parameters.txt"
ParPathRun = "results/parameters/4000"
Paths=[PrecPath, Evap_Path, TempPath, FlowAccPath, FlowDPath, ]

#p2=[1, 227.31]
p2=[24, 1530]
init_st=[0,5,5,5,0]
snow = 0

st, q_out, q_uz_routed, q_lz_trans = RunModel(HBV,Paths,ParPathRun,p2,init_st,snow)

#%% store the result into rasters
# create list of names 
src=gdal.Open(FlowAccPath)
s=dt.datetime(2012,06,14,19,00,00)
e=dt.datetime(2013,12,23,00,00,00)
index=pd.date_range(s,e,freq="1H")
resultspath="results/"
names=[resultspath+str(i)[:-6] for i in index]
names=[i.replace("-","_") for i in names]
names=[i.replace(" ","_") for i in names]
names=[i+".tif" for i in names]

#GIS.RastersLike(src,q_uz_routed[:,:,:-1],names)
