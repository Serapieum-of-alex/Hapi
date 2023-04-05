"""Created on Thu Nov 14 13:42:10 2019.

@author: mofarrag
This code reads the SWIM output file (.dat file) that contains the time series
of discharge for some computational nodes and calculate some statistical properties

the code assumes that the time series are of a daily temporal resolution, and
that the hydrological year is 1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

Inputs:
    1-path:
        [string] path where the files exist
    2-ObservedFile:
        [string] the name of the SWIM result file (the .dat file)
    3-ComputationalNodesFile:
        [string] the name of the file which contains the ID of the computational
        nodes you want to do the statistical analysis for, the ObservedFile
        should contain the discharge time series of these nodes in order
    4-WarmUpPeriod:
        [integer] the number of days you want to neglect at the begining of the
        Simulation (warm up period)
    5-StartDate:
        [string] the begining date of the time series,
    6-saveto:
        [string ] the path where you want to  save the statistical properties
Outputs:
    1-Statistical Properties.csv:
        file containing some statistical properties like mean, std, min, 5%, 25%,
        median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
        q100, q200, q500
"""
import os

CompP = r"F:\01Algorithms\Hydrology\HAPI"
os.chdir(CompP)
import Hapi.hm.calibration as RC
import Hapi.hm.inputs as IN

# import datetime as dt

GaugesF = "examples/Hydrodynamic models/test_case/inputs/gauges/gauges.csv"
Calib = RC.Calibration("RIM", version=3)
Calib.readGaugesTable(GaugesF)

# path = CompP + "/base_data/Calibration/"
# ObservedFile = "GRDC"
WarmUpPeriod = 0
start = "1955-1-1"
TSdirectory = "examples/Hydrodynamic models/test_case/inputs/gauges/discharge_long_ts/"
saveto = TSdirectory + "/statistical-analysis-results/"
SavePlots = True
NoValue = -9
#%%
"""
create the DistributionProperties.csv & Statistical Properties.csv files with
"""
Inputs35 = IN.Inputs("Observed_Q")

computationalnodes = Calib.hm_gauges["oid"].tolist()

Inputs35.statistical_properties(
    computationalnodes,
    TSdirectory,
    start,
    WarmUpPeriod,
    SavePlots,
    saveto,
    SeparateFiles=True,
    Filter=NoValue,
    method="lmoments",
    file_extension=".csv",
)
#%% using gumbel
Inputs35 = IN.Inputs("Observed_Q")

Inputs35.read_xs(
    "examples/Hydrodynamic models/test_case/inputs/1d/topo/xs_same_downward-3segment.csv"
)
computationalnodes = Calib.hm_gauges["oid"].tolist()

Inputs35.statistical_properties(
    computationalnodes,
    TSdirectory,
    start,
    WarmUpPeriod,
    SavePlots,
    saveto,
    SeparateFiles=True,
    Filter=NoValue,
    Distibution="GUM",
    method="lmoments",
    file_extension=".csv",
)
#%% for the results
TSdirectory = "examples/Hydrodynamic models/test_case/results/customized_results/discharge_long_ts/"
saveto = TSdirectory + "/statistical-analysis-results/"
Inputs35 = IN.Inputs("HM_results")
Inputs35.read_xs(
    "examples/Hydrodynamic models/test_case/inputs/1d/topo/xs_same_downward-3segment.csv"
)

Inputs35.statistical_properties(
    Inputs35.segments,
    TSdirectory,
    start,
    WarmUpPeriod,
    SavePlots,
    saveto,
    SeparateFiles=True,
    Filter=NoValue,
    Distibution="GUM",
    method="lmoments",
    file_extension=".txt",
    Results=True,
)
