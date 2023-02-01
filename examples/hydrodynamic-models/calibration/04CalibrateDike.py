"""Created on Sat Apr  4 20:57:53 2020.

@author: mofarrag
"""
# from IPython import get_ipython
# get_ipython().magic('reset -f')
import os

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import pandas as pd
# from matplotlib import animation``
# import datetime as dt
# import math
# from Hapi.sm import performancecriteria as Pf
import Hapi.hm.river as R
from Hapi.hm.event import Event as E

# import Hapi.Visualizer as V
#%% Paths
CompP = r"F:\01Algorithms\Hydrology\HAPI"
os.chdir(CompP)
rpath = "examples/Hydrodynamic models/test_case/"
# rpath = r"F:\02Case-studies"
# CompP = rpath + r"\ClimXtreme\rim_base_data\setup\rhine"
# os.chdir(CompP)
oldxsPath = rpath + r"inputs/1d/topo/xs_same_downward-3segment.csv"
# newxsPath = CompP + r"\inputs\1d\topo\xs_elevated.csv"

start = "1955-01-01"
days = 21910
River = R.River("RIM", version=3, days=days, start=start)
# River.OneDResultPath = wpath + "/results/1d/"
# RIM2Files = "F:/02Case studies/Rhine/base_data/Calibration/RIM2.0/01 calibrated parameters/06-15042020/"

River.readXS(oldxsPath)
River.readSlope(rpath + "/inputs/1d/topo/slope.csv")
River.OneDResultPath = rpath + "/results/1d/"

River.readRiverNetwork(rpath + "/inputs/1d/topo/rivernetwork-3segments.txt")
# River.ReturnPeriod(CompP1 + "/base_data/HYDROMOD/RIM1.0/Observed Period/Statistical Analysis/" + "HQRhine.csv")
path = r"examples/Hydrodynamic models/test_case/results/customized_results/discharge_long_ts/Statistical analysis results/DistributionProperties.csv"
River.statisticalProperties(path)
#%%

# calculate the capacity of the bankfull area
River.getCapacity("Qbkf")
# calculate the capacity of the whole cross section till the lowest dike level
River.getCapacity("Qc2", Option=2)

River.calibrateDike("RP", "QcRP")
River.cross_sections["ZlDiff"] = (
        River.cross_sections["zlnew"].values - River.cross_sections["zl"].values
)
River.cross_sections["ZrDiff"] = (
        River.cross_sections["zrnew"].values - River.cross_sections["zr"].values
)
# River.cross_sections.to_csv(RIM2Files+"XS100.csv", index = None)

#%%
# read the overtopping files
# River.Overtopping()
# Event object
Event = E.Event("RIM2.0")
Event.overtopping(wpath + "/processing/" + "overtopping.txt")
# get the end days of each event
Event.getAllEvents()

River.EventIndex = Event.event_index
# read the left and right overtopping 1D results
River.overtopping()

XSleft = list()
XSright = list()
print("No of Events = " + str(len(Event.end_days)))
for i in range(len(Event.end_days)):
    # get the cross sectin that was overtopped for a specific day
    XSlefti, XSrighti = River.getOvertoppedXS(Event.end_days[i], True)
    XSleft = XSleft + XSlefti
    XSright = XSright + XSrighti

XSright = list(set(XSright))
XSleft = list(set(XSleft))
XSleft.sort()
XSright.sort()
# raise the left dike of the overtopped cross section by 0.5 meter
for i in XSleft:
    # print(i)
    # print(River.cross_sections.loc[i-1,'xsid'])
    River.cross_sections.loc[i - 1, "zl"] = River.cross_sections.loc[i - 1, "zl"] + 0.5

for i in XSright:
    # print(i)
    # print(River.cross_sections.loc[i-1,'xsid'])
    River.cross_sections.loc[i - 1, "zr"] = River.cross_sections.loc[i - 1, "zr"] + 0.5

# get the subs that was inundated
# floodedSubs = River1.GetFloodedSubs(OvertoppedXS = XSleft + XSright)

#%% Save the new cross section file
River.cross_sections.to_csv(newxsPath, index=None)
