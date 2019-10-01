# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:03:14 2018

@author: Mostafa
"""
#%links

import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/trials")

import sys
#sys.path.append("")

#%library
import numpy as np
import matplotlib.pyplot as plt
import pysheds
import gdal


import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pysheds.grid import Grid
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# functions


#%% read DEM
DEM = Grid.from_raster('DEM/n30w100_con/n30w100_con', data_name='dem')
#grid = Grid.from_raster('DEM/dem_2000.tif', data_name='dem')
#fig, ax = plt.subplots(figsize=(8,6))
#fig.patch.set_alpha(0)
#
#plt.imshow(DEM.dem, extent=DEM.extent, cmap='cubehelix', zorder=1)
#plt.colorbar(label='Elevation (m)')
#plt.grid(zorder=0)
#plt.title('Digital elevation map')
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.tight_layout()
#plt.savefig('img/conditioned_dem.png', bbox_inches='tight')
#%% 
flow_dir=Grid.flowdir(DEM.dem, data=None, out_name='dir', nodata_in=None, nodata_out=0,
                pits=-1, flats=-1, dirmap= (64,  128,  1,   2,    4,   8,    16,  32),
                inplace=True, mask=False, ignore_metadata=False)
f=Grid._set_dirmap(DEM.dem,(64,  128,  1,   2,    4,   8,    16,  32), dirr)
Grid.fi
#%%plot
plt.figure(1,figsize=(15,8))
plt.plot()

plt.xlabel("")
plt.ylabel("")
plt.legend([""])

