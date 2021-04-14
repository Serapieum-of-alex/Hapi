# Distributed Hydrological Model

After preparing all the meteorological, GIS inputs required for the model, and Extracting the parameters for the catchment 

## 1- Catchment Object
- Import the Catchment object which is the main object in the distributed model, to read and check the input data,  and when the model finish the simulation it stores the results and do the visualization


```

	class Catchment():

	    ======================
	       Catchment
	    ======================
	    Catchment class include methods to read the meteorological and Spatial inputs
	    of the distributed hydrological model. Catchment class also reads the data
	    of the gauges, it is a super class that has the run subclass, so you
	    need to build the catchment object and hand it as an inpit to the Run class
	    to run the model

	    methods:
	        1-ReadRainfall
	        2-ReadTemperature
	        3-ReadET
	        4-ReadFlowAcc
	        5-ReadFlowDir
	        6-ReadFlowPathLength
	        7-ReadParameters
	        8-ReadLumpedModel
	        9-ReadLumpedInputs
	        10-ReadGaugeTable
	        11-ReadDischargeGauges
	        12-ReadParametersBounds
	        13-ExtractDischarge
	        14-PlotHydrograph
	        15-PlotDistributedQ
	        16-SaveResults

	    def __init__(self, name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
	                 TemporalResolution = "Daily"):
	        =============================================================================
	            Catchment(name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
	                             TemporalResolution = "Daily")
	        =============================================================================
	        Parameters
	        ----------
	        name : [str]
	            Name of the Catchment.
	        StartDate : [str]
	            starting date.
	        EndDate : [str]
	            end date.
	        fmt : [str], optional
	            format of the given date. The default is "%Y-%m-%d".
	        SpatialResolution : TYPE, optional
	            Lumped or 'Distributed' . The default is 'Lumped'.
	        TemporalResolution : TYPE, optional
	            "Hourly" or "Daily". The default is "Daily".
```

- To instantiate the object you need to provide the `name`, `statedate`, `enddate`, and the `SpatialResolution`

```
		from Hapi.catchment import Catchment

		start = "2009-01-01"
		end = "2011-12-31"
		name = "Coello"

		Coello = Catchment(name, start, end, SpatialResolution = "Distributed")
```

## Read Meteorological Inputs

- First define the directory where the data exist

```
			PrecPath = "Hapi/Data/00inputs/meteodata/4000/calib/prec-CPC-NOAA" #
			Evap_Path = "Hapi/Data/00inputs/meteodata/4000/calib/evap"
			TempPath = "Hapi/Data/00inputs/meteodata/4000/calib/temp"
			FlowAccPath = "Hapi/Data/00inputs/GIS/4000/acc4000.tif"
			FlowDPath = "Hapi/Data/00inputs/GIS/4000/fd4000.tif"
			ParPathRun = "Hapi/Model/results/parameters/02lumped parameters/Parameter set-1/"
			SaveTo = "Hapi/Model/results/"
```

- Then use the each method in the object to read the coresponding data

```
			Coello.ReadRainfall(PrecPath)
			Coello.ReadTemperature(TempPath)
			Coello.ReadET(Evap_Path)
			Coello.ReadFlowAcc(FlowAccPath)
			Coello.ReadFlowDir(FlowDPath)
```

- To read the parameters you need to provide whether you need to consider the snow subroutine or not

```
			Snow = 0
			Coello.ReadParameters(ParPathRun, Snow)
```

### 2- Lumped Model
- Get the Lumpde conceptual model you want to couple it with the distributed routing module which in our case HBV 
	and define the initial condition, and catchment area.
```
			import Hapi.hbv_bergestrom92 as HBV

			CatchmentArea = 1530
			InitialCond = [0,5,5,5,0]
			Coello.ReadLumpedModel(HBV, CatchmentArea, InitialCond)
```
- If the Inpus are consistent in dimensions you will get a the following message

![check_inputs](../img/check_inputs.png)

- to check the performance of the model we need to read the gauge hydrographs

```
Coello.ReadGaugeTable("Hapi/Data/00inputs/Discharge/stations/gauges.csv", FlowAccPath)
GaugesPath = "Hapi/Data/00inputs/Discharge/stations/"
Coello.ReadDischargeGauges(GaugesPath, column='id', fmt="%Y-%m-%d")
```

### 3-Run Object

- The `Run` object connects all the components of the simulation together, the `Catchment` object, the `Lake` object and the `distributedrouting` object
- import the Run object and use the `Catchment` object as a parameter to the `Run` object, then call the RunHapi method to start the simulation

```
			from Hapi.run import Run
			Run.RunHapi(Coello)
```

- the result of the simulation will be stored as attributes in the Catchment object as follow

Outputs:
    1-statevariables: [numpy attribute]
        4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
    2-qlz: [numpy attribute]
        3D array of the lower zone discharge
    3-quz: [numpy attribute]
        3D array of the upper zone discharge
    4-qout: [numpy attribute]
        1D timeseries of discharge at the outlet of the catchment
        of unit m3/sec
    5-quz_routed: [numpy attribute]
        3D array of the upper zone discharge  accumulated and
        routed at each time step
    6-qlz_translated: [numpy attribute]
        3D array of the lower zone discharge translated at each time step

### 4-Extract Hydrographs

- The final step is to extract the simulated Hydrograph from the cells at the location of the gauges to compare
- The `ExtractDischarge` method extracts the hydrographs, however you have to provide in the gauge file the coordinates of the gauges with the same coordinate system of the `FlowAcc` raster

```
			Coello.ExtractDischarge(Factor=Coello.GaugesTable['area ratio'].tolist())

			for i in range(len(Coello.GaugesTable)):
			    gaugeid = Coello.GaugesTable.loc[i,'id']
			    print("----------------------------------")
			    print("Gauge - " +str(gaugeid))
			    print("RMSE= " + str(round(Coello.Metrics.loc['RMSE',gaugeid],2)))
			    print("NSE= " + str(round(Coello.Metrics.loc['NSE',gaugeid],2)))
			    print("NSEhf= " + str(round(Coello.Metrics.loc['NSEhf',gaugeid],2)))
			    print("KGE= " + str(round(Coello.Metrics.loc['KGE',gaugeid],2)))
			    print("WB= " + str(round(Coello.Metrics.loc['WB',gaugeid],2)))
			    print("Pearson CC= " + str(round(Coello.Metrics.loc['Pearson-CC',gaugeid],2)))
			    print("R2 = " + str(round(Coello.Metrics.loc['R2',gaugeid],2)))
```

- The `ExtractDischarge` will print the performance metics


### 5-Visualization

- Firts type of visualization we can do with the results is to compare the gauge hydrograph with the simulatied hydrographs 
- Call the `PlotHydrograph` method and provide the period you want to visualize with the order of the gauge

```
		gaugei = 5
		plotstart = "2009-01-01"
		plotend = "2011-12-31"

		Coello.PlotHydrograph(plotstart, plotend, gaugei)
```

![hydrograph](../img/hydrograph.png) 

