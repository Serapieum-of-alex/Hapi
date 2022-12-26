"""This code is written to calculate the performance metrics for the simulated hydrographs using the Hydaulic model.

- To run this code you have to prepare the calibration data first in a specific format
and folder structure and to do that you have to run
the code 01CalibrationDataPreparation.py""
"""
import datetime as dt

#%% Libraries
import os

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import Hapi.hm.calibration as RC

"""change directory to the processing folder inside the project folder"""
os.chdir(r"C:\MyComputer\01Algorithms\Hydrology\Hapi")
rpath = os.path.abspath(os.getcwd() + "/examples/Hydrodynamic models/test_case")
saveto = rpath
#%% Links
Comp = rpath + r"\ClimXtreme\rim_base_data\setup"
rrm_path = rf"{rpath}\inputs\rrm"
hm_q_path = rf"{rpath}\results\separated-results\discharge"
hmwl_path = rf"{rpath}\results\separated-results\water_level"
hm_files = rpath + "/inputs/1d/topo/"
SaveTo = Comp + "/base_data/calibration_results/"
# %% gauges
gauges_file = rpath + "/inputs/gauges/gauges.csv"
wl_obs_path = rpath + "/inputs/gauges/water_level/"
q_obs_path = rpath + "/inputs/gauges/discharge/"

novalue = -9
start = "1955-01-01"
end = "1955-03-21"
Calib = RC.Calibration("HM", start=start)
Calib.readGaugesTable(gauges_file)
Calib.readObservedQ(q_obs_path, start, end, novalue, gauge_date_format="'%Y-%m-%d'")
Calib.readObservedWL(wl_obs_path, start, end, novalue, gauge_date_format="'%Y-%m-%d'")
# sort the gauges table based on the segment
Calib.hm_gauges.sort_values(by="id", inplace=True, ignore_index=True)
#%% Read RIM results
Calib.readHMQ(hm_q_path, fmt="'%Y-%m-%d'")
Calib.ReadHMWL(hmwl_path, fmt="'%Y-%m-%d'")

Calib.readRRM(
    f"{rrm_path}/hm_location",
    fmt="'%Y-%m-%d'",
    location=2,
    path2=f"{rrm_path}/rrm_location",
)
Calib.readRiverNetwork(hm_files + "/rivernetwork-3segments.txt")
#%% calculate metrics
Calib.HMvsRRM()  # start ="1990-01-01"
# mHM vs observed
Calib.RRMvsObserved()  # start ="1990-01-01"
# GRDC vs RIM
Calib.HMQvsObserved()  # start ="1990-01-01"
# Water levels GRDC vs RIM
Calib.HMWLvsObserved()  # start ="1990-01-01"
#%% plotting Hydrographs
subid = 1
gaugei = 0
# start ="1990-01-01"
start = ""
end = ""  # "1994-3-1"
summary, fig, ax = Calib.InspectGauge(subid, gaugei=gaugei, start=start, end=end)
print(summary)
#%% special plot for the poster
subid = 1
gaugei = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == subid, Calib.gauge_id_col].values[
    0
]

fromdate = dt.datetime(1955, 1, 1)
todate = dt.datetime(1955, 3, 21)
i = Calib.rrm_gauges.index(gaugei)
# for i in range(len(Calib.rrm_gauges)):
plt.figure(int(Calib.rrm_gauges[i]), figsize=(10, 8))

plt.plot(
    Calib.q_rrm.loc[fromdate:todate, Calib.q_rrm.columns[i]],
    label="SWIM",
    linewidth=3,
    linestyle="-",
)  # , color ="#DC143C"

plt.plot(
    Calib.q_gauges[Calib.q_gauges.columns[i]].loc[fromdate:todate],
    label="Observed Hydrograph",
    linewidth=3,
    linestyle="dashed",
)  # , color = "#DC143C"
plt.plot(
    Calib.q_hm[Calib.q_hm.columns[i]].loc[fromdate:todate],
    label="RIM",
    linewidth=3,
    linestyle="-.",
)  # , color = "green"
# SimMax = max(Calib.q_hm[Calib.q_hm.columns[i]].loc[Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'start']:Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'end']])
# ObsMax = max(Calib.q_rrm[Calib.q_rrm.columns[i]].loc[Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'start']:Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'end']])
# pos = max(SimMax, ObsMax)

# plt.annotate("SubID = " + str(int(Calib.q_hm.columns[i])), xy=(dt.datetime(1971,1,1),pos-10),
#             fontsize = 20)
# plt.annotate("RMSE = " + str(round(Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'rmse'],2)), xy=(dt.datetime(1971,1,1),pos-40),
#             fontsize = 15)
# plt.annotate("KGE = " + str(round(Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'KGE'],2)), xy=(dt.datetime(1971,1,1),pos-70),
#             fontsize = 15)
# plt.annotate("NSE = " + str(round(Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'NSE'],2)), xy=(dt.datetime(1971,1,1),pos-100),
#             fontsize = 15)
# plt.annotate("WB = " + str(round(Calib.MetricsHM_RRM.loc[Calib.rrm_gauges[i],'WB'],2)), xy=(dt.datetime(1971,1,1),pos-130),
#             fontsize = 15)
plt.xlabel("Time", fontsize=15)
plt.ylabel("Discharge m3/s", fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=15, framealpha=0.1)
plt.title("Station: " + str(subid), fontsize=30)
plt.tight_layout()
# plt.savefig(str(subid)+".tif", transparent=True)
#%% save Metrics dataframe to display in arc map
Calib.SaveMetices(SaveTo)
# sumarry.to_csv(DataPath + "summary.txt")
