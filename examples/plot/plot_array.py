import os

# os.chdir("F:/01Algorithms/Hydrology/HAPI/examples")
import matplotlib

matplotlib.use("TkAgg")
import pandas as pd
from osgeo import gdal

from Hapi.plot.map import Map

# %% Paths
RasterAPath = "examples/data/GIS/Hapi_GIS_Data/dem_100_f.tif"
RasterBPath = "examples/data/GIS/Hapi_GIS_Data/acc4000.tif"
pointsPath = "examples/GIS/data/points.csv"
"""
to plot the array you need to read the raster using gdal
"""
# read the raster
src = gdal.Open(RasterAPath)
#%%
"""
then using all the default parameters in the PlotArray method you can directly plot the gdal.Dataset
"""
Map.PlotArray(src)
"""
However as you see in the plot you might need to adjust the color to different color scheme or the
display of the colorbar, colored label.
you might don't need to display the labels showing the values of each cell, and for all of these
decisions there are a lot of customizable parameters
"""
#%% options
"""
first for the size of the figure you have to pass a tuple with the width and height
Figsize : [tuple], optional
        figure size. The default is (8,8).
Title : [str], optional
        title of the plot. The default is 'Total Discharge'.
titlesize : [integer], optional
        title size. The default is 15.
"""
Figsize = (8, 8)
Title = "DEM"
titlesize = 15

Map.PlotArray(src, Figsize=Figsize, Title=Title, titlesize=titlesize)
#%%
"""color bar

Cbarlength : [float], optional
        ratio to control the height of the colorbar. The default is 0.75.
orientation : [string], optional
        orintation of the colorbar horizontal/vertical. The default is 'vertical'.
cbarlabelsize : integer, optional
        size of the color bar label. The default is 12.
cbarlabel : str, optional
        label of the color bar. The default is 'Discharge m3/s'.
rotation : [number], optional
        rotation of the colorbar label. The default is -90.
TicksSpacing : [integer], optional
        Spacing in the colorbar ticks. The default is 2.
"""
Cbarlength = 0.75
orientation = "vertical"
cbarlabelsize = 12
cbarlabel = "Elevation"
rotation = -80
TicksSpacing = 500
"""
rotation : [number], optional
    rotation of the colorbar label. The default is -90.
"""
Map.PlotArray(
    src,
    Cbarlength=Cbarlength,
    orientation=orientation,
    cbarlabelsize=cbarlabelsize,
    cbarlabel=cbarlabel,
    rotation=rotation,
    TicksSpacing=TicksSpacing,
)
# %%
"""
color schame

ColorScale : integer, optional
    there are 5 options to change the scale of the colors. The default is 1.
    1- ColorScale 1 is the normal scale
    2- ColorScale 2 is the power scale
    3- ColorScale 3 is the SymLogNorm scale
    4- ColorScale 4 is the PowerNorm scale
    5- ColorScale 5 is the BoundaryNorm scale
    ------------------------------------------------------------------
    gamma : [float], optional
        value needed for option 2 . The default is 1./2..
    linthresh : [float], optional
        value needed for option 3. The default is 0.0001.
    linscale : [float], optional
        value needed for option 3. The default is 0.001.
    midpoint : [float], optional
        value needed for option 5. The default is 0.
    ------------------------------------------------------------------
cmap : [str], optional
    color style. The default is 'coolwarm_r'.
"""
# for normal linear scale
ColorScale = 1
cmap = "terrain"
Map.PlotArray(src, ColorScale=ColorScale, cmap=cmap, TicksSpacing=TicksSpacing)
# %%
# for power scale
"""
the more you lower the value of gamma the more of the color bar you give to the lower value range
"""
ColorScale = 2
gamma = 0.5

Map.PlotArray(
    src,
    ColorScale=ColorScale,
    cmap=cmap,
    gamma=gamma,
    TicksSpacing=TicksSpacing,
    Title=f"gamma = {gamma}",
)
# %%
gamma = 0.4
Map.PlotArray(
    src,
    ColorScale=ColorScale,
    cmap=cmap,
    gamma=gamma,
    TicksSpacing=TicksSpacing,
    Title=f"gamma = {gamma}",
)
# %%
gamma = 0.2
Map.PlotArray(
    src,
    ColorScale=ColorScale,
    cmap=cmap,
    gamma=gamma,
    TicksSpacing=TicksSpacing,
    Title=f"gamma = {gamma}",
)
# %%  SymLogNorm scale
ColorScale = 3
linscale = 0.001
linthresh = 0.0001
Map.PlotArray(
    src,
    ColorScale=ColorScale,
    linscale=linscale,
    linthresh=linthresh,
    cmap=cmap,
    TicksSpacing=TicksSpacing,
)
# %% PowerNorm scale
ColorScale = 4
Map.PlotArray(src, ColorScale=ColorScale, cmap=cmap, TicksSpacing=TicksSpacing)
# %% color scale 5
ColorScale = 5
midpoint = 20
Map.PlotArray(
    src, ColorScale=ColorScale, midpoint=midpoint, cmap=cmap, TicksSpacing=TicksSpacing
)
# %%
src = gdal.Open(RasterBPath)
arr = src.ReadAsArray()
# %% PowerNorm scale
ColorScale = 4
TicksSpacing = 10
Map.PlotArray(src, ColorScale=ColorScale, cmap=cmap, TicksSpacing=TicksSpacing)
# %%
"""
Cell value label

display_cellvalue : [bool]
    True if you want to display the values of the cells as a text
NumSize : integer, optional
    size of the numbers plotted intop of each cells. The default is 8.
Backgroundcolorthreshold : [float/integer], optional
    threshold value if the value of the cell is greater, the plotted
    numbers will be black and if smaller the plotted number will be white
    if None given the maxvalue/2 will be considered. The default is None.

"""
display_cellvalue = True
NumSize = 8
Backgroundcolorthreshold = None

Map.PlotArray(
    src,
    display_cellvalue=display_cellvalue,
    NumSize=NumSize,
    Backgroundcolorthreshold=Backgroundcolorthreshold,
    TicksSpacing=TicksSpacing,
)
# %%
"""
if you have points that you want to display in the map you can read it into a dataframe
in condition that it has two columns "x", "y" which are the coordinates of the points of theand they have to be
in the same coordinate system as the raster
"""

# read the points
points = pd.read_csv(pointsPath)

Gaugecolor = "blue"
Gaugesize = 100
IDcolor = "green"
IDsize = 20
Map.PlotArray(
    src,
    Gaugecolor=Gaugecolor,
    Gaugesize=Gaugesize,
    IDcolor=IDcolor,
    IDsize=IDsize,
    points=points,
    display_cellvalue=display_cellvalue,
    NumSize=NumSize,
    Backgroundcolorthreshold=Backgroundcolorthreshold,
    TicksSpacing=TicksSpacing,
)
