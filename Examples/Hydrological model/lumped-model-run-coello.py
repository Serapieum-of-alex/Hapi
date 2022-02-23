import datetime as dt

import matplotlib

matplotlib.use('TkAgg')
import Hapi.rrm.hbv_bergestrom92 as HBVLumped
import Hapi.sm.performancecriteria as PC
from Hapi.catchment import Catchment
from Hapi.rrm.routing import Routing
from Hapi.run import Run

# %% data
Parameterpath = "Examples/Hydrological model/data/lumped_model/Coello_Lumped2021-03-08_muskingum.txt"
MeteoDataPath = "Examples/Hydrological model/data/lumped_model/meteo_data-MSWEP.csv"
Path = "Examples/Hydrological model/data/lumped_model/"
SaveTo = "Examples/Hydrological model/data/lumped_model/"
### Meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.ReadLumpedInputs(MeteoDataPath)
# %% ### Lumped model

# catchment area
AreaCoeff = 1530
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.ReadLumpedModel(HBVLumped, AreaCoeff, InitialCond)
# %% ### Model Parameters

Snow = 0  # no snow subroutine
Coello.ReadParameters(Parameterpath, Snow)
# Coello.Parameters
# %% ### Observed flow
Coello.ReadDischargeGauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
# %%  ### Routing

# RoutingFn = Routing.TriangularRouting2
RoutingFn = Routing.Muskingum_V
Route = 1
# %% ### Run The Model
Run.RunLumped(Coello, Route, RoutingFn)
# %% ### Calculate performance criteria
Metrics = dict()

# gaugeid = Coello.QGauges.columns[-1]
Qobs = Coello.QGauges['q']

Metrics['RMSE'] = PC.RMSE(Qobs, Coello.Qsim['q'])
Metrics['NSE'] = PC.NSE(Qobs, Coello.Qsim['q'])
Metrics['NSEhf'] = PC.NSEHF(Qobs, Coello.Qsim['q'])
Metrics['KGE'] = PC.KGE(Qobs, Coello.Qsim['q'])
Metrics['WB'] = PC.WB(Qobs, Coello.Qsim['q'])

print("RMSE= " + str(round(Metrics['RMSE'], 2)))
print("NSE= " + str(round(Metrics['NSE'], 2)))
print("NSEhf= " + str(round(Metrics['NSEhf'], 2)))
print("KGE= " + str(round(Metrics['KGE'], 2)))
print("WB= " + str(round(Metrics['WB'], 2)))
# %% ### Plot Hydrograph
gaugei = 0
plotstart = "2009-01-01"
plotend = "2011-12-31"
Coello.PlotHydrograph(plotstart, plotend, gaugei, Title="Lumped Model")
# %% ### Save Results

StartDate = "2009-01-01"
EndDate = "2010-04-20"

Path = SaveTo + "Results-Lumped-Model_" + str(dt.datetime.now())[0:10] + ".txt"
Coello.SaveResults(Result=5, start=StartDate, end=EndDate, Path=Path)
