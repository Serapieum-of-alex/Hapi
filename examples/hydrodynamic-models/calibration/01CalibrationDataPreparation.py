"""This code is written to prepare the data for the calibration comparison and performance calculation first it search for the reaches that exist in GRDC data then it collects the RIM results for these specific sub-basins from different files into one folder "RIM", the "RIMResultsPath" should have all the folders containing the separated rim results called by the period of the run (ex, 1000-2000, 5000-6000), plus a folder called.

it also extract the coresponding GRDC observed data from GRDC file and store them
by the subID in a folder called "GRDC"

The code also extract the simulated hydrograph using swim at the down stream node
to the sub-basins
"""
import os
import matplotlib
matplotlib.use("TkAgg")
rpath = r"C:\gdrive\\Case-studies"
Comp = rf"{rpath}\ClimXtreme\rim_base_data\setup"
os.chdir(Comp + "/base_data/calibration_results")
import Hapi.hm.calibration as RC
import Hapi.hm.river as R
#%% Hydraulic model files
hm_results_path = rf"{rpath}\ClimXtreme\rim_base_data\setup\freq_analysis_rhine\1\results\1d"
hm_data = f"{Comp}/base_data/calibrated_cross_sections/rhine/"

base_dir = rf"{rpath}\ClimXtreme\rim_base_data\setup\freq_analysis_rhine\1\results\gauges_results"
save_q = f"{base_dir}/q/"
save_wl = f"{base_dir}/wl/"
#%% Gauges data
gauges_dir = f"{rpath}/ClimXtreme/data"
GaugesF = f"{gauges_dir}/gauges/rhine_gauges.geojson"

novalue = -9
start = "1951-01-01"
end = "2003-12-31"

Calib = RC.Calibration("RIM", version=3)
Calib.readGaugesTable(GaugesF)
#%%
start = "1955-1-1"
rrmstart = "1955-1-1"
River = R.River("RIM", version=3, start=start, rrmstart=rrmstart)
River.onedresultpath = hm_results_path
River.readSlope(hm_data + "/slope_rhine.csv")
River.readXS(hm_data + "/xs_rhine.csv")
#%%
column = "oid"
segments = list(set(Calib.hm_gauges['id']))

for i in range(20, len(segments)):
    SubID = segments[i]
    if not os.path.exists(f"{hm_results_path}/{SubID}.zip"):
        print(f"{hm_results_path}/{SubID}.zip file does not exist")
        continue
    Sub = R.Reach(SubID, River)
    Sub.read1DResult(path=hm_results_path, extension=".zip")
    # get the gauges that are in the segment
    Gauges = Calib.hm_gauges.loc[Calib.hm_gauges['id'] == SubID, :]
    Gauges.index = range(len(Gauges))
    for j in range(len(Gauges)):
        GagueXS = Gauges.loc[j, 'xsid']
        if column == "oid" or column == "qid":
            fname = Gauges.loc[j, column]
        else:
            fname = str(SubID) + "_" + str(GagueXS)

        # Extract Results at the gauges
        Sub.read1DResult(xsid=GagueXS)
        print("Extract the XS results - " + str(fname))
        Q = Sub.xs_hydrograph[GagueXS].resample('D').mean().to_frame()
        Q["date"] = ["'" + str(i)[:10] + "'" for i in Q.index]
        Q = Q.loc[:, ['date', GagueXS]]
        WL = Sub.xs_water_level[GagueXS].resample('D').mean().to_frame()
        WL["date"] = Q["date"]
        WL = WL.loc[:, ['date', GagueXS]]
        Q.to_csv(f"{save_q}{fname}.txt", index=False, index_label='Date', float_format="%.3f")
        WL.to_csv(f"{save_wl}{fname}.txt", index=False, index_label='Date', float_format="%.3f")

