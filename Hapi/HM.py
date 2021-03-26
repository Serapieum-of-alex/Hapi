# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:27:13 2021

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic('reset -f')
import os
# import datetime as dt
import pandas as pd
import numpy as np
import Hapi.river as R
from Hapi.interface import Interface

Comp = "F:/RFM/mHM2RIM_testcase"
os.chdir(Comp + "/RIM/processing")
#%% Paths
# the working directory of the project
wpath = Comp + "/RIM"
RRMPath = Comp + "/base_data/calibration/mHM"
RIM2Files = wpath + "/inputs/1d/topo/"
BaseDataPath = Comp + "/base_data"
savepath = wpath + "/results/customized_results/"

start = "1952-1-1"
RRMstart = "1952-1-1"

River = R.River('RIM', Version=3, start=start, RRMstart=RRMstart, RRMdays = 23192)

River.OneDResultPath = Comp + "/base_data/calibration/calibration_results/all_results/20210315/"
River.USbndPath = wpath + "/results/USbnd/"
River.OneMinResultPath = wpath + "/results/"
River.TwoDResultPath = wpath + "/results/2d/zip/"
River.CustomizedRunsPath = wpath + "/results/customized_results/"
River.Compressed = True
River.RRMPath = RRMPath

Path = wpath + "/processing/def1D.txt"
River.Read1DConfigFile(Path)

# River.Slope(RIM2Files + "/slope.csv")
# River.ReadCrossSections(RIM2Files + "/XS.csv")
# River.RiverNetwork(RIM2Files + "/rivernetwork.txt")
#%%
IF = Interface('Rhine')
IF.ReadCrossSections(RIM2Files + "/XS.csv")
IF.RiverNetwork(RIM2Files + "/rivernetwork.txt")
IF.ReadLateralsTable(wpath + "/inputs/1d/topo/laterals.txt")
IF.ReadLaterals(Path=wpath + "/inputs/1d/hydro/", date_format='%d_%m_%Y')
IF.ReadBoundaryConditionsTable(wpath + "/inputs/1d/topo/BonundaryConditions.txt")
IF.ReadBoundaryConditions(Path = wpath + "/inputs/1d/hydro/", date_format='%d_%m_%Y')
#%% Sub-basin
""" Write the Sub-ID you want to visualize its results """

River.RoutedQ = np.zeros(shape=(River.NoTimeSteps,River.NoSeg))
# sum of all US routedQ
River.DirectUS = np.zeros(shape=(River.NoTimeSteps,River.NoSeg))
# sum of the DirectUS and BC
River.TotalUS = np.zeros(shape=(River.NoTimeSteps,River.NoSeg))

HydrologicalTempRes = 24
OneMinTimeSteps = 60 * HydrologicalTempRes


SubID = 2
Sub = R.Sub(SubID,River)
Sub.GetFlow(IF, SubID)
# HQ : is a rating curve table contains discharge at the first column and coresponding
#water depth at the second column
# HQ is allocated inside the RatingCurve subroutine
Sub.GetRatingCurve()
# get the area and perimeters of the cross section if the water level is at 
# min of the intermediate point left and right and then the max of both
Sub.GetXSGeometry()
Sub.GetUSHydrograph(River)
    
for i in range(River.SimStart, River.SimEnd+1):
    print("Step-" + str*(i))
    
    
    
    


#%%