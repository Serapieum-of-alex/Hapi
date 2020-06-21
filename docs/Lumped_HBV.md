# HBV Lumped conceptual model
The HBV model [Bergström, 1992] is usually run with daily time steps, but higher resolution (hourly) can be used if data are available. Input data are precipitation, air temperature and potential evapotranspiration.

HBV model consists of three main components:
snow accumulation and melt, soil moisture accounting, response and river routing subroutines

![HBV Component](../img/water_cycle.png)
[Bergström, 1992]

The model has number of free parameters, values of which are found by calibration, There are also parameters describing the characteristics of the basin and its climate

The soil moisture accounting calculations require data on the potential evapotranspiration.Normally monthly mean standard values are sufficient, but more detailed data can also
be used. The source of these data may either be calculations according to the Penman formula or similar, or measurements by evaporimeters. In the latter case it is important
to correct for systematic errors before entering the model.

where 14 parameters 

## Snow
The snow routine controls snow accumulation and melt. The precipitation accumulates as snow when the air temperature drops below a threshold value (TT). snow accumulation is adjusted by a free parameter, Sfcf, the snowfall correction factor.
Melt starts with temperatures above the threshold, TT, according to a simple degree-day
```
MELT = Cfmax * (T - TT)
where: MELT = snowmelt (mm/day)
Cfmax = degree-day factor (mm/°C · day)
TT = temperature threshold (C).
```
The liquid water holding capacity of snow has to be exceeded before any runoff is generated. It is usually preset to 10 % . A refreezing coefficient, which is used to refreeze free water in the snow if snowmelt is interrupted, is fix.ed in the code.
Thus the snow routine of the HBV model has primarily three free parameters that have to be estimated by calibration: TT, CsF and CMELT· lf a separation into vegetation zones is used, the number doubles. It is also common to use separate threshold temperatures
for snow accumulation and melt. 
The snow routine of the HBV model has been subject to major modifications in the Norwegian, Finnish and Swiss versions of the model. A statistical routine for redistribution of snow over the timber-line has, for example, been introduced by Killingtveit and Aam (1978), and several attempts have been made to introduce glacier-melt subroutines 

["tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"]
