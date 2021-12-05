Comp = "F:/01Algorithms/Hydrology/HAPI/Examples"
import datetime as dt

import pandas as pd

import Hapi.rrm.hbv_bergestrom92 as HBVLumped
import Hapi.sm.performancecriteria as PC
from Hapi.rrm.calibration import Calibration
from Hapi.rrm.routing import Routing
from Hapi.run import Run

#%% Paths
Parameterpath = Comp + "/data/lumped/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = Comp + "/data/lumped/meteo_data-MSWEP.csv"
Path = Comp + "/data/lumped/"
#%% Meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"

Coello = Calibration(name, start, end)
Coello.ReadLumpedInputs(MeteoDataPath)
#%% Basic_inputs
# catchment area
AreaCoeff = 1530
# temporal resolution
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]
# no snow subroutine
Snow = 0
Coello.ReadLumpedModel(HBVLumped, AreaCoeff, InitialCond)
#%% Calibration parameters
# Calibration boundaries
UB = pd.read_csv(Path + "/UB-3.txt", index_col=0, header=None)
parnames = UB.index
UB = UB[1].tolist()
LB = pd.read_csv(Path + "/LB-3.txt", index_col=0, header=None)
LB = LB[1].tolist()

Maxbas = True
Coello.ReadParametersBounds(UB, LB, Snow, Maxbas=Maxbas)
#%% additional arguments
parameters = []
# Routing
Route = 1
RoutingFn = Routing.TriangularRouting1

Basic_inputs = dict(Route=Route, RoutingFn=RoutingFn, InitialValues=parameters)
#%%
### Objective function
# outlet discharge
Coello.ReadDischargeGauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")

OF_args = []
OF = PC.RMSE

Coello.ReadObjectiveFn(PC.RMSE, OF_args)
#%% Optimization
"""
API options
Create the options dictionary all the optimization parameters should be passed
to the optimization object inside the option dictionary:


to see all options import Optimizer class and check the documentation of the
method setOption

"""
ApiObjArgs = dict(
    hms=100,
    hmcr=0.95,
    par=0.65,
    dbw=2000,
    fileout=1,
    xinit=0,
    filename=Path + "/Lumped_History" + str(dt.datetime.now())[0:10] + ".txt",
)

for i in range(len(ApiObjArgs)):
    print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

# pll_type = 'POA'
pll_type = None

ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=True, hot_start=False)

OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]
#%%
# run calibration
cal_parameters = Coello.LumpedCalibration(
    Basic_inputs, OptimizationArgs, printError=None
)

print("Objective Function = " + str(round(cal_parameters[0], 2)))
print("Parameters are " + str(cal_parameters[1]))
print("Time = " + str(round(cal_parameters[2]["time"] / 60, 2)) + " min")
#%% run the model
Coello.Parameters = cal_parameters[1]
Run.RunLumped(Coello, Route, RoutingFn)
#%% calculate performance criteria
Metrics = dict()

Qobs = Coello.QGauges[Coello.QGauges.columns[0]]

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
#%% plotting
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.PlotHydrograph(plotstart, plotend, gaugei, Title="Lumped Model")

#%% save the parameters
ParPath = Path + "Parameters" + str(dt.datetime.now())[0:10] + ".txt"
parameters = pd.DataFrame(index=parnames)
parameters["values"] = cal_parameters[1]
parameters.to_csv(ParPath, header=None, float_format="%0.4f")

#%% Save Results
StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = Path + "Results-Lumped-Model" + str(dt.datetime.now())[0:10] + ".txt"
Coello.SaveResults(Result=5, StartDate=StartDate, EndDate=EndDate, Path=Path)
