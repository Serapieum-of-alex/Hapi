# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:30:48 2020

@author: mofarrag

"""
from IPython import get_ipython
get_ipython().magic("reset -f")
from Hapi.raster import Raster
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Paths
FileName = "F:/01Algorithms/Hydrology/HAPI/Examples/data/GIS/MSWEP_1979010100.nc"
SaveTo = "F:/01Algorithms/Hydrology/HAPI/Examples/data/GIS/"
VarName=None
#%%
Raster.NCtoTiff(FileName, SaveTo)

#%% plot

src = rasterio.open(SaveTo + "MSWEP_1979010100.nc")
fig = plt.figure(figsize=(12,8))
im = plt.imshow(src.read(1)/100.0, cmap="gist_rainbow")
plt.title("Monthly mean sea level pressure")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)
plt.tight_layout()
plt.show()