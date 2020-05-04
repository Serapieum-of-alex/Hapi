# -*- coding: utf-8 -*-
"""
This code is used to run the lumpedmodel

-   you have to make the root directory to the examples folder to enable the code
    from reading input files
- Example catchment needs to be calibrated

"""
#%library
#import numpy as np
import pandas as pd

# Hapi modules
import Hapi.hbvlumped as HBVLumped
import Hapi.run as RUN
from Hapi.routing import RoutingMAXBAS
#import Hapi.GISpy as GIS
#import GISCatchment as GC
#from Hapi.RUN import RunModel
#import Hapi.HBV as HBV
#%%
### meteorological data
data=pd.read_csv("data/lumped/meteo_data.txt",header=0 ,delimiter=',',#"\t", #skiprows=11,
                   engine='python',index_col=0)
data_matrix=data.as_matrix()

### Basic_inputs
ConceptualModel=HBVLumped
p2=[24, 1530]
init_st=[0,5,5,5,0]
snow = 0

### parameters
"""
First model needs to be calibrated to have the model parameters

"""
parameters= []#np.loadtxt("")

### Routing
Routing=1
RoutingFn=RoutingMAXBAS
### run the model
st, q_sim=RUN.RunLumped(ConceptualModel,data_matrix,parameters,p2,init_st,snow,Routing, RoutingFn)
#%% store the result into rasters
# create list of names