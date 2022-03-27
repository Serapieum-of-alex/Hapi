"""
This code is used to Run the distributed model for jiboa rover in El Salvador
wher the catchment is consisted os a ustream lake and a volcanic area
-   you have to make the root directory to the examples folder to enable the code
    from reading input files

"""
import os

root = "E:\case studies\El Salvador"
os.chdir(root)
import datetime as dt

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import Hapi.rrm.hbv as HBV
import Hapi.rrm.hbv_lake as HBVLake
import Hapi.sm.performancecriteria as Pf
from Hapi.catchment import Catchment, Lake
from Hapi.run import Run

# %% Paths
res = 4000
"""
paths to meteorological data
"""
PrecPath = "inputs/Hapi/meteodata/" + str(res) + "/calib/prec_clipped"
Evap_Path = "inputs/Hapi/meteodata/" + str(res) + "/calib/evap_clipped"
TempPath = "inputs/Hapi/meteodata/" + str(res) + "/calib/temp_clipped"
FlowAccPath = "inputs/Hapi/GIS/" + str(res) + "_matched/acc" + str(res) + ".tif"
FlowDPath = "inputs/Hapi/GIS/" + str(res) + "_matched/fd" + str(res) + ".tif"
ParPath = "inputs/Hapi/meteodata/" + str(res) + "/parameters/"
# Lake
LakeMeteoPath = "inputs/Hapi/meteodata/lakedata.csv"
LakeParametersPath = "inputs/Hapi/meteodata/" + str(res) + "/Lakeparameters.txt"
GaugesPath = "inputs/Hapi/meteodata/Gauges/"
SaveTo = "results/"
# %% Distributed Model Object

CatchmentArea = 227.31
InitialCond = np.loadtxt("inputs/Hapi/meteodata/Initia-jiboa.txt", usecols=0).tolist()
Snow = 0

start_date = "2012-06-14 19:00:00"
# Edate = '2014-11-17 00:00:00'
end_date = "2013-12-23 00:00:00"
name = "Jiboa"
Jiboa = Catchment(
    name,
    start_date,
    end_date,
    SpatialResolution="Distributed",
    TemporalResolution="Hourly",
    fmt="%Y-%m-%d %H:%M:%S",
)
Jiboa.ReadRainfall(PrecPath)
Jiboa.ReadTemperature(TempPath)
Jiboa.ReadET(Evap_Path)
Jiboa.ReadFlowAcc(FlowAccPath)
Jiboa.ReadFlowDir(FlowDPath)
Jiboa.ReadParameters(ParPath, Snow)

Jiboa.ReadLumpedModel(HBV, CatchmentArea, InitialCond)
# %% Lake Object
"""
lake meteorological data
"""
# where the lake discharges its flow (give the indices of the cell)
if res == 4000:
    OutflowCell = [2, 1]  # 4km
elif res == 2000:
    OutflowCell = [4, 2]  # 2km
elif res == 1000:
    OutflowCell = [10, 4]  # 1km
elif res == 500:
    OutflowCell = [19, 10]  # 500m

start_date = "2012.06.14 19:00:00"
# Edate = '2014.11.17 00:00:00'
end_date = "2013.12.23 00:00:00"

JiboaLake = Lake(
    start=start_date,
    end=end_date,
    fmt="%Y.%m.%d %H:%M:%S",
    TemporalResolution="Hourly",
    Split=True,
)

JiboaLake.ReadMeteoData(LakeMeteoPath, fmt="%d.%m.%Y %H:%M")
JiboaLake.ReadParameters(LakeParametersPath)

StageDischargeCurve = np.loadtxt("inputs/Hapi/meteodata/curve.txt")
LakeInitCond = np.loadtxt("inputs/Hapi/meteodata/Initia-lake.txt", usecols=0).tolist()
LakeCatArea = 133.98
LakeArea = 70.64
Snow = 0
JiboaLake.ReadLumpedModel(
    HBVLake, LakeCatArea, LakeArea, LakeInitCond, OutflowCell, StageDischargeCurve, Snow
)
# %% Gauges
Date1 = "14.06.2012 19:00"
Date2 = "23.12.2013 00:00"
Jiboa.ReadGaugeTable(GaugesPath + "GaugesTable.csv", FlowAccPath)
Jiboa.ReadDischargeGauges(
    GaugesPath, column="id", fmt="%d.%m.%Y %H:%M", Split=True, Date1=Date1, Date2=Date2
)
# %% run the model
Run.RunHAPIwithLake(Jiboa, JiboaLake)
# %% calculate some metrics
Jiboa.ExtractDischarge(OnlyOutlet=True)

for i in range(len(Jiboa.GaugesTable)):
    gaugeid = Jiboa.GaugesTable.loc[i, "id"]
    print("----------------------------------")
    print("Gauge - " + str(gaugeid))
    print("RMSE= " + str(round(Jiboa.Metrics.loc["RMSE", gaugeid], 2)))
    print("NSE= " + str(round(Jiboa.Metrics.loc["NSE", gaugeid], 2)))
    print("NSEhf= " + str(round(Jiboa.Metrics.loc["NSEhf", gaugeid], 2)))
    print("KGE= " + str(round(Jiboa.Metrics.loc["KGE", gaugeid], 2)))
    print("WB= " + str(round(Jiboa.Metrics.loc["WB", gaugeid], 2)))
    print("Pearson CC= " + str(round(Jiboa.Metrics.loc["Pearson-CC", gaugeid], 2)))
    print("R2 = " + str(round(Jiboa.Metrics.loc["R2", gaugeid], 2)))
# %%
Qobs = Jiboa.QGauges[Jiboa.GaugesTable.loc[0, "id"]]

gaugeid = Jiboa.GaugesTable.loc[0, "id"]

WS = {}
WS["type"] = 1
WS["N"] = 3
ModelMetrics = dict()
ModelMetrics["Calib_RMSEHF"] = round(
    Pf.RMSEHF(Qobs, Jiboa.Qsim[gaugeid], WS["type"], WS["N"], 0.75), 3
)
ModelMetrics["Calib_RMSELF"] = round(
    Pf.RMSELF(Qobs, Jiboa.Qsim[gaugeid], WS["type"], WS["N"], 0.75), 3
)
ModelMetrics["Calib_NSEHf"] = round(Pf.NSE(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_NSELf"] = round(
    Pf.NSE(np.log(Qobs), np.log(Jiboa.Qsim[gaugeid])), 3
)
ModelMetrics["Calib_RMSE"] = round(Pf.RMSE(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_KGE"] = round(Pf.KGE(Qobs, Jiboa.Qsim[gaugeid]), 3)
ModelMetrics["Calib_WB"] = round(Pf.WB(Qobs, Jiboa.Qsim[gaugeid]), 3)

print(ModelMetrics)
# %% plot
gaugei = 0
plotstart = "2012-06-16"
plotend = "2013-12-23"

Jiboa.PlotHydrograph(plotstart, plotend, gaugei)
# %%
"""
=============================================================================
PlotDistributedResults(StartDate, EndDate, fmt="%Y-%m-%d", Option = 1, Gauges=False,
                    TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
                    NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None,
                    cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
                    Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
                    Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
                    linscale=0.001, midpoint=0, orientation='vertical', rotation=-90,
                    **kwargs):
=============================================================================
PlotDistributedResults animate the time series of the meteorological inputs and
the result calculated by the model  like the total discharge, upper zone,
and lower zone discharge and the state variables

Parameters
----------
StartDate : [str]
    starting date
EndDate : [str]
    end date
fmt : [str]
    format of the gicen date. The default is "%Y-%m-%d"
Option : [str]
    1- Total discharge, 2-Upper zone discharge, 3-ground water,
    4-Snowpack state variable, 5-Soil moisture, 6-Upper zone,
    7-Lower zone, 8-Water content, 9-Precipitation input. 10-ET,
    11-Temperature. The default is 1
Gauges : [str]
    . The default is False
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

plotstart = "2012-07-20"
plotend = "2012-08-20"

Anim = Jiboa.PlotDistributedResults(
    plotstart,
    plotend,
    Figsize=(8, 8),
    Option=3,
    threshold=160,
    PlotNumbers=False,
    TicksSpacing=10,
    Interval=10,
    Gauges=False,
    cmap="inferno",
    Textloc=[0.6, 0.8],
    Gaugecolor="red",
    ColorScale=2,
    IDcolor="blue",
    IDsize=25,
    gamma=0.08,
)
# %%
Path = SaveTo + "anim.mov"
Jiboa.SaveAnimation(VideoFormat="mov", Path=Path, SaveFrames=3)
# %% Save Results
StartDate = "2012-07-20"
EndDate = "2012-08-20"

Path = SaveTo + "Lumped_Parameters_" + str(dt.datetime.now())[0:10] + "_"
Jiboa.SaveResults(
    Result=1, StartDate=StartDate, EndDate=EndDate, Path=Path, FlowAccPath=FlowAccPath
)
