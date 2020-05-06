# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:41:18 2018

@author: Mostafa
"""
#library
import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Model/colombia")
import sys
sys.path.append("C:/Users/Mostafa/Desktop/My Files/Research/Hapi")
path="C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Data/colombia/00inputs/"

import numpy as np
import pandas as pd

#import Hapi.GISpy as GIS
#import GISCatchment as GC
from Hapi.RUN import RunModel
import Hapi.HBV as HBV
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

p2=[24, 1530]
init_st=[0,5,5,5,0]
snow = 0

_, q_out, _, _ = RunModel(HBV,Paths,ParPathRun,p2,init_st,snow)


par=["rfcf", "fc", "beta","etf","lp","cflux", "k","k1","alpha","perc","kmusk","Xmusk"]

parameters=np.loadtxt("parameters.txt", usecols=0).tolist()
UB=np.loadtxt("UB.txt", usecols=0).tolist()
LB=np.loadtxt("LB.txt", usecols=0).tolist()

parameter2=pd.DataFrame(index=par)
parameter2['value']=parameters