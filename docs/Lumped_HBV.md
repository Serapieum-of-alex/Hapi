# HBV Lumped conceptual model
The SMHI version of the HBV model is usually run with daily time steps, but higher resolution (hourly) can be used if data are available. Input data are precipitation, air temperature and potential evapotranspiration.

HBV model consists of three main components:
snow accumulation and melt, soil moisture accounting, response and river routing subroutines

![HBV Component](../img/water_cycle.tif)
The model has number of free parameters, values of which are found by calibration, There are also parameters describing the characteristics of the basin and its climate

The soil moisture accounting calculations require data on the potential evapotranspiration.Normally monthly mean standard values are sufficient, but more detailed data can also
be used. The source of these data may either be calculations according to the Penman formula or similar, or measurements by evaporimeters. In the latter case it is important
to correct for systematic errors before entering the model.


The Lumped conceptual model used in Hapi is HBV [Bergstr€om, 1992]
where 14 parameters 

["tt","sfcf","cfmax","cwh","cfr","fc","beta","lp","k0","k1","k2","uzl","perc","maxbas"]
