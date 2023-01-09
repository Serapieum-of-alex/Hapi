"""
This code is written to prepare the data for the calibration comparison
and performance calculation
first it search for the reaches that exist in GRDC data then it collects
the RIM results for these specific sub-basins from different files
into one folder "RIM", the "RIMResultsPath" should have all the folders
containing the separated rim results called by the period of the run
(ex, 1000-2000, 5000-6000), plus a folder called

it also extract the coresponding GRDC observed data from GRDC file and store them
by the subID in a folder called "GRDC"

The code also extract the simulated hydrograph using swim at the down stream node
to the sub-basins

Inputs:
    1- DataPath :
        [string] path to the folder where the calibration folders ("GRDC","RIM",
        "SWIM") the GRDC data, Trace file, RIMSubBasinFile exist
    2- GRDC data:
            1- GRDC_file:
                [string] .dat file contains time series of daily discharge data
                with ID associated with each one
            2-processed_data_file:
                [string] Excel file contains station names with column "MessID"
                having the an ID and anothe column "SWIMBasinID" having the
                Sub-basin ID

    3- Trace file:
        [string] text file contains two columns the first in the sub-basin ID and
        the second is the ID of the down stream computational swim node

    4- RIMSubBasinFile:
        [string] text file contains the Sub-basin IDs which is routed using RIM


files names have to contain the extension (".txt", ".xls",".xlxs")

Outputs:
    1- "GRDC","RIM","SWIM" :
        txt file for each sub-basin with the name as the subID in three folders
        "GRDC","RIM","SWIM"
    2- "calibratedSubs.txt" :
        text file contains the subID of the sub-basins which have GRDC data and
        those are the sub-basins that the code collects their data in the
        previous three folders
"""
#%% Libraries
import os
import matplotlib
matplotlib.use('TkAgg')
#Comp = "F:/02Case studies/Rhine"
Comp = r"C:\gdrive\Case-studies\ClimXtreme\rim_base_data\setup"
os.chdir(Comp + "/base_data/calibration_results")
import Hapi.hm.calibration as RC
import Hapi.hm.river as R
#%% Links
# RIM files
RIMResultsPath = Comp + "/base_data/calibration_results/all_results/rhine/"
RRMPath = Comp + "/base_data/mHM/"

DataPath = Comp + "/base_data/calibration_results/"
RIMdata = Comp + "/base_data/calibrated_cross_sections/rhine/"
SaveQ = DataPath + "/gauge_results/Discharge/"
SaveWL = DataPath + "/gauge_results/Water_Level/"
GaugesPath = "F:/RFM/ClimXtreme/data"

addHQ2 = False
SaveTo = Comp + "/base_data/calibration/"
#%% Gauges data

GaugesF = GaugesPath + "/gauges/rhine_gauges.geojson"
WLGaugesPath = GaugesPath + "/gauges/Water_Levels/"
QgaugesPath = GaugesPath + "/gauges/discharge/"

novalue = -9
start = "1951-01-01"
end = "2003-12-31"

Calib = RC.Calibration("RIM", version=3)
Calib.readGaugesTable(GaugesF)
Calib.readObservedQ(QgaugesPath, start, end, novalue)
Calib.readObservedWL(WLGaugesPath, start, end, novalue)
#%%
start = "1955-1-1"
rrmstart = "1955-1-1"

River = R.River('RIM', version=3, start=start, rrmstart=rrmstart)
River.onedresultpath = RIMResultsPath
River.readSlope(RIMdata + "/slope_rhine.csv")
River.readXS(RIMdata + "/xs_rhine.csv")
# River.RiverNetwork(RIMdata + "/rivernetwork.txt")
#%%
column = "oid"
segments = list(set(Calib.hm_gauges['id']))
for i in range(len(segments)):
    SubID = segments[i]
    Sub = R.Sub(SubID,River)
    Sub.read1DResult()
    # get the gauges that are in the segment
    Gauges = Calib.hm_gauges.loc[Calib.hm_gauges['id'] == SubID, :]
    Gauges.index = range(len(Gauges))
    for j in range(len(Gauges)):
        GagueXS = Gauges.loc[j,'xsid']
        if column == "oid" or column == "qid":
            fname = Gauges.loc[j,column]
        else:
            fname = str(SubID) + "_" + str(GagueXS)
            
        # Extract Results at the gauges
        Sub.read1DResult(xsid=GagueXS)
        print("Extract the XS results - " + str(fname))
        # Q = Sub.XSHydrographs[GagueXS].to_frame()#.resample('D').mean()
        Q = Sub.XSHydrographs[GagueXS].resample('D').mean().to_frame()
        Q["date"] = ["'" + str(i)[:10] + "'" for i in Q.index]
        Q = Q.loc[:,['date',GagueXS]]
        WL = Sub.XSWaterLevel[GagueXS].resample('D').mean().to_frame()
        WL["date"]= Q["date"]
        WL = WL.loc[:,['date',GagueXS]]
        Q.to_csv(SaveQ + str(fname) +".txt", index = False, index_label='Date', float_format="%.3f")
        WL.to_csv(SaveWL + str(fname) +".txt", index = False, index_label='Date', float_format="%.3f")