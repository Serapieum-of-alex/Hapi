
# Hapi Lumped Model

to run the HBV lumped model inside Hapi you need to prepare the meteorological inputs (rainfall, temperature and potential evapotranspiration), HBV parameters, and the HBV model (you can load Bergström, 1992 version of HBV from Hapi )

- First load the prepared lumped version of the HBV module inside Hapi, the triangular routing function and the wrapper function that runs the lumped model `RUN`.
```
import Hapi.hbv_bergestrom92 as HBVLumped
import Hapi.run as RUN
from Hapi.routing import TriangularRouting
```
- read the meteorological data, data has be in the form of numpy array with the following order [rainfall, ET, Temp, Tm], ET is the potential evapotranspiration, Temp is the temperature (C), and Tm is the long term monthly average temperature.
```
import numpy as np
import pandas as pd

data=pd.read_csv("meteo_data.txt",header=0 ,delimiter=',', index_col=0)
data=data.values
```
- Loat the pre-estimated parameters with the following order ["rfcf","tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"] if the catchment has snow, if not ["rfcf","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"] and convert it to list

```
parameters = pd.read_csv("parameter.txt", index_col = 0, header = None)
parameters = parameters[1].tolist()
```
- prepare the initial conditions, snow option (if you want to simulate snow accumulation and snow melt or not), temporal resolution, and cathcment area.
```
### Basic_inputs
ConceptualModel = HBVLumped
# p2 = [temporal resolution, catchment area]
p2=[24, 1530]
init_st=[0,10,10,10,0]
# no snow subroutine
snow = 0
```
- prepare the routing options (whether you want to route the generated discharge or not, if yes the routing function).
```
### Routing
Routing=1
RoutingFn=TriangularRouting
```
- now all the data required for the model are prepared in the right form, now you can call the `RunLumped` wrapper to initiate the calculation
```
st, q_sim=RUN.RunLumped(ConceptualModel,data,parameters,p2,init_st,snow,Routing, RoutingFn)
```
the `RunLumped` returns two numpy arrays first is the state variables [snow pack, soil moisture, upper zone, lower zone, water content], and second array is the calculated discharge.

to calculate some metrics for the quality assessment of the calculate discharge the `performancecriteria` contains some metrics like `RMSE`, `NSE`, `KGE` and `WB` , you need to load it, a measured time series of doscharge for the same period of the simulation is also needed for the comparison.

all methods in `performancecriteria` takes two numpy arrays of the same length and return real number.
```
import Hapi.performancecriteria as PC

# observed flow
Qobs =np.loadtxt("measuredQ.txt")

Metrics = dict()

Metrics['RMSE'] = PC.RMSE(Qobs, q_sim)
Metrics['NSE'] = PC.NSE(Qobs, q_sim)
Metrics['NSEhf'] = PC.NSEHF(Qobs, q_sim)
Metrics['KGE'] = PC.KGE(Qobs, q_sim)
Metrics['WB'] = PC.WB(Qobs, q_sim)
```
to plot the calculated and measured discharge import matplotlib

```
import matplotlib.pyplot as plt

plt.figure(1, figsize=(12,8))
plt.plot(q_sim)
plt.plot(Qobs)
plt.xlabel("Time (daily)")
plt.ylabel("Flow Hydrograph m3/s")
```