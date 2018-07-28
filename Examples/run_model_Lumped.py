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
# data
path="C:/Users/Mostafa/Desktop/My Files/Research/Data_and_Models/Data/colombia/00inputs/"

#%library
import numpy as np
#import gdal
#import datetime as dt
import pandas as pd
#import matplotlib.pyplot as plt

# functions
import HBVLumped
#import Wrapper
import RUN
from Routing import RoutingMAXBAS
#import Hapi.GISpy as GIS
#import GISCatchment as GC
#from Hapi.RUN import RunModel
#import Hapi.HBV as HBV
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

### parameters
parameters= []#np.loadtxt("")

### Routing
Routing=1
RoutingFn=RoutingMAXBAS
### run the model
st, q_sim=RUN.runLumped(ConceptualModel,data,parameters,p2,init_st,snow,Routing, RoutingFn)
#%% store the result into rasters
# create list of names 