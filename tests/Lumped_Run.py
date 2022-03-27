"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
import datetime as dt

Comp = "F:/01Algorithms/Hydrology/HAPI/examples"
#%% Modules
import Hapi.rrm.hbv_bergestrom92 as HBVLumped
import Hapi.sm.performancecriteria as PC
from Hapi.catchment import Catchment
from Hapi.rrm.routing import Routing
from Hapi.run import Run

#%% Paths
Parameterpath = Comp + "/data/lumped/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = Comp + "/data/lumped/meteo_data-MSWEP.csv"
Path = Comp + "/data/lumped/"
#%%
### meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.ReadLumpedInputs(MeteoDataPath)
#%%
### Basic_inputs
# catchment area
AreaCoeff = 1530
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.ReadLumpedModel(HBVLumped, AreaCoeff, InitialCond)

### parameters
Snow = 0  # no snow subroutine
Coello.ReadParameters(Parameterpath, Snow)
#%% observed flow
Coello.ReadDischargeGauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
#%% Routing
# RoutingFn = Routing.TriangularRouting2
RoutingFn = Routing.Muskingum_V
Route = 1
### run the model
Run.RunLumped(Coello, Route, RoutingFn)
#%% calculate performance criteria
Metrics = dict()

Qobs = Coello.QGauges["q"]

Metrics["RMSE"] = PC.RMSE(Qobs, Coello.Qsim["q"])
Metrics["NSE"] = PC.NSE(Qobs, Coello.Qsim["q"])
Metrics["NSEhf"] = PC.NSEHF(Qobs, Coello.Qsim["q"])
Metrics["KGE"] = PC.KGE(Qobs, Coello.Qsim["q"])
Metrics["WB"] = PC.WB(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(Metrics["RMSE"], 2)))
print("NSE= " + str(round(Metrics["NSE"], 2)))
print("NSEhf= " + str(round(Metrics["NSEhf"], 2)))
print("KGE= " + str(round(Metrics["KGE"], 2)))
print("WB= " + str(round(Metrics["WB"], 2)))
#%% Plot
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.PlotHydrograph(plotstart, plotend, gaugei, Title="Lumped Model")
#%% Save Results
StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = Path + "Results-Lumped-Model" + str(dt.datetime.now())[0:10] + ".txt"
Coello.SaveResults(Result=1, StartDate=StartDate, EndDate=EndDate, Path=Path)
