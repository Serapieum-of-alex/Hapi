"""Created on Sun Jun 24 21:02:34 2018.

@author: Mostafa
"""
#%links
# from IPython import get_ipython   # to reset the variable explorer each time
# get_ipython().magic('reset -f')
import os

Comp = "F:/02Case-studies/"
# Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "/Coello/Hapi/Model/")

import Hapi.rrm.hbv_bergestrom92 as HBV
from Hapi.catchment import Catchment

# import numpy as np
from Hapi.run import Run

#%% Paths
path = Comp + "/Coello/Hapi/Data/00inputs/"
# PrecPath = path + "meteodata/4000/calib/prec" #
PrecPath = path + "meteodata/4000/calib/prec-CPC-NOAA"  #
# PrecPath = path + "meteodata/4000/calib/prec-MSWEP" #
Evap_Path = path + "meteodata/4000/calib/evap"
TempPath = path + "meteodata/4000/calib/temp"
FlowAccPath = path + "GIS/4000/acc4000.tif"
FlowDPath = path + "GIS/4000/fd4000.tif"
ParPathRun = "results/parameters/02lumped parameters/Parameter-set-Avg/"
# ParPathRun = "results/parameters/00default parameters/min/"
SaveTo = "results/saved rasters/"
#%% Flood model data
RiverNetworkF = path + "GIS/4000/river_network.tif"
BankfullDepthF = path + "GIS/4000/bankfulldepth.tif"
RiverWidthF = path + "GIS/4000/river_width.tif"
RiverRoughnessF = path + "GIS/4000/channel_roughness.tif"
FloodPlainRoughnessF = path + "GIS/4000/floodplain_roughness.tif"
DEMF = path + "GIS/4000/dem4000.tif"
RouteRiver = "Kinematic"
#%% Meteorological data
AreaCoeff = 1530
InitialCond = [0, 5, 5, 5, 0]
Snow = 0
"""
Create the model object and read the input data
"""
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(
    name, start, end, SpatialResolution="Distributed", RouteRiver=RouteRiver
)


Coello.readFlowAcc(FlowAccPath)
Coello.readFlowDir(FlowDPath)
Coello.ReadRiverGeometry(
    DEMF, BankfullDepthF, RiverWidthF, RiverRoughnessF, FloodPlainRoughnessF
)

Coello.readParameters(ParPathRun, Snow)
Coello.readRainfall(PrecPath)
Coello.readTemperature(TempPath)
Coello.readET(Evap_Path)
Coello.readLumpedModel(HBV, AreaCoeff, InitialCond)
#%% Gauges
Coello.readGaugeTable(path + "Discharge/stations/gauges.csv", FlowAccPath)
GaugesPath = path + "Discharge/stations/"
Coello.readDischargeGauges(GaugesPath, column="id", fmt="%Y-%m-%d")
#%% Run the model
"""
Outputs:
    ----------
    1-statevariables: [numpy attribute]
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute]
        3D array of the lower zone discharge
    3-quz: [numpy attribute]
        3D array of the upper zone discharge
    4-qout: [numpy attribute]
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute]
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step
"""
Run.RunFloodModel(Coello)
#%% calculate performance criteria
Coello.extractDischarge(Factor=Coello.GaugesTable["area ratio"].tolist())

for i in range(len(Coello.GaugesTable)):
    gaugeid = Coello.GaugesTable.loc[i, "id"]
    print("----------------------------------")
    print("Gauge - " + str(gaugeid))
    print("RMSE= " + str(round(Coello.Metrics.loc["RMSE", gaugeid], 2)))
    print("NSE= " + str(round(Coello.Metrics.loc["NSE", gaugeid], 2)))
    print("NSEhf= " + str(round(Coello.Metrics.loc["NSEhf", gaugeid], 2)))
    print("KGE= " + str(round(Coello.Metrics.loc["KGE", gaugeid], 2)))
    print("WB= " + str(round(Coello.Metrics.loc["WB", gaugeid], 2)))
    print("Pearson CC= " + str(round(Coello.Metrics.loc["Pearson-CC", gaugeid], 2)))
    print("R2 = " + str(round(Coello.Metrics.loc["R2", gaugeid], 2)))
#%% plot
gaugei = 5
plotstart = "2009-01-01"
plotend = "2011-12-31"

Coello.plotHydrograph(plotstart, plotend, gaugei)
#%%
"""
=============================================================================
AnimateArray(Arr, Time, NoElem, TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
       NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None,
       cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
       Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
       Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
       linscale=0.001, midpoint=0, orientation='vertical', rotation=-90,IDcolor = "blue",
          IDsize =10, **kwargs)
=============================================================================
Parameters
----------
Arr : [array]
    the array you want to animate.
Time : [dataframe]
    dataframe contains the date of values.
NoElem : [integer]
    Number of the cells that has values.
TicksSpacing : [integer], optional
    Spacing in the colorbar ticks. The default is 2.
Figsize : [tuple], optional
    figure size. The default is (8,8).
PlotNumbers : [bool], optional
    True to plot the values intop of each cell. The default is True.
NumSize : integer, optional
    size of the numbers plotted intop of each cells. The default is 8.
Title : [str], optional
    title of the plot. The default is 'Total Discharge'.
titlesize : [integer], optional
    title size. The default is 15.
Backgroundcolorthreshold : [float/integer], optional
    threshold value if the value of the cell is greater, the plotted
    numbers will be black and if smaller the plotted number will be white
    if None given the maxvalue/2 will be considered. The default is None.
textcolors : TYPE, optional
    Two colors to be used to plot the values i top of each cell. The default is ("white","black").
cbarlabel : str, optional
    label of the color bar. The default is 'Discharge m3/s'.
cbarlabelsize : integer, optional
    size of the color bar label. The default is 12.
Cbarlength : [float], optional
    ratio to control the height of the colorbar. The default is 0.75.
Interval : [integer], optional
    number to controlthe speed of the animation. The default is 200.
cmap : [str], optional
    color style. The default is 'coolwarm_r'.
Textloc : [list], optional
    location of the date text. The default is [0.1,0.2].
Gaugecolor : [str], optional
    color of the points. The default is 'red'.
Gaugesize : [integer], optional
    size of the points. The default is 100.
IDcolor : [str]
    the ID of the Point.The default is "blue".
IDsize : [integer]
    size of the ID text. The default is 10.
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
orientation : [string], optional
    orintation of the colorbar horizontal/vertical. The default is 'vertical'.
rotation : [number], optional
    rotation of the colorbar label. The default is -90.
**kwargs : [dict]
    keys:
        Points : [dataframe].
            dataframe contains two columns 'cell_row', and cell_col to
            plot the point at this location

Returns
-------
animation.FuncAnimation.

"""

plotstart = "2009-01-01"
plotend = "2009-02-01"

Anim = Coello.plotDistributedResults(
    plotstart,
    plotend,
    Figsize=(9, 9),
    Option=1,
    threshold=160,
    PlotNumbers=True,
    TicksSpacing=5,
    Interval=200,
    Gauges=True,
    cmap="inferno",
    Textloc=[0.1, 0.2],
    Gaugecolor="red",
    ColorScale=1,
    IDcolor="blue",
    IDsize=25,
)

#%%
Path = SaveTo + "anim.gif"
Coello.saveAnimation(VideoFormat="gif", Path=Path, SaveFrames=3)
#%% Save the result into rasters

StartDate = "2009-01-01"
EndDate = "2010-04-20"
Prefix = "Qtot_"

Coello.saveResults(
    FlowAccPath,
    Result=1,
    StartDate=StartDate,
    EndDate=EndDate,
    Path="F:/02Case studies/",
    Prefix=Prefix,
)
# Hapi/Model/results/Coello/
