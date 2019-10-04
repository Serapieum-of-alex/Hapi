# -*- coding: utf-8 -*-
"""
Created on Sun May 06 20:11:56 2018

@author: Mostafa
"""
#%links
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model")

import sys
sys.path.append("HBV_distributed/function")

#%library
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

import shapely as sh
# functions
import plotting_functions as plf
#%% 

lumped_catch = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data'), "Lumped_catchment.shp")
boundary=gpd.read_file(lumped_catch)  # polygon geometry type
boundary['geometry'].to_crs(epsg=4326)
data=boundary
data=data.append(boundary,ignore_index=True)

#geom=boundary['geometry']
#coord_type="x"
#%%
data=plf.XY(data)

indf = boundary
outdf = gpd.GeoDataFrame() #columns=indf.columns
for idx, row in indf.iterrows():
#        if type(row.geometry) == Polygon:
#            outdf = outdf.append(row,ignore_index=True)
#        if type(row.geometry) == MultiPolygon:
    multdf = gpd.GeoDataFrame() #columns=indf.columns
    recs = len(row.geometry)
    multdf = multdf.append([row]*recs,ignore_index=True)
    for geom in range(recs):
        multdf.loc[geom,'geometry'] = row.geometry[geom]
    outdf = outdf.append(multdf,ignore_index=True)
#    return outdf

#%%
roads_fp = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data/Example'), "roads.shp")
fp = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data/Example'), "TravelTimes_to_5975375_RailwayStation.shp")
metro_fp = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data/Example'), "metro.shp")

# read the data 
data=gpd.read_file(fp)        # polygon geometry type
roads=gpd.read_file(roads_fp) # lineString geometry type
metro=gpd.read_file(metro_fp) # 
#%%
geom_type=[]
data=metro
type1=["Point","LineString","LinearRing","Polygon","MultiPoint",
       "MultiLineString","MultiPolygon","GeometryCollection"]
# road multiLineString 39,158
# data multiLineString 39,158
for i in range(len(data)):
    if data['geometry'][i].type ==type1[6]:
        print(i)
        
#np.where(metro['geometry'][i].type =="MultiPolygon")

#%%
# change the coordinate system 
#data['geometry'].to_crs(epsg=3067)
#metro['geometry'].to_crs(epsg=3067)
#roads['geometry'].to_crs(epsg=3067)

# functions 
#
""" calculate the x & y coordinates of the grid (polygon)"""
# apply function on all rows of dataframe
data1=plf.XY(data)

data['x']=data.apply(plf.getCoords,geom_col="geometry", coord_type="x", axis=1)
data['y']=data.apply(plf.getCoords,geom_col="geometry", coord_type="y", axis=1)

"""Calculate the x and y coordinates of the roads (these contain MultiLineStrings) """
roads1=plf.XY(roads)

roads['x']=roads.apply(plf.getCoords,geom_col="geometry", coord_type="x", axis=1)
roads['y']=roads.apply(plf.getCoords,geom_col="geometry", coord_type="y", axis=1)

""" Calculate the x and y coordinates of metro. """
metro1=plf.XY(metro)
metro['x']=metro.apply(plf.getCoords,geom_col="geometry", coord_type="x", axis=1)
metro['y']=metro.apply(plf.getCoords,geom_col="geometry", coord_type="y", axis=1)


#%plot
