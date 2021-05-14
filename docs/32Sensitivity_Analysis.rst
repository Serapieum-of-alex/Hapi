******************************
Sensetivity Analysis (OAT)
******************************
OAT sensitivity analysis is a tool that is based 

One of the simplest and most common approaches of sensitivity analysis is that of changing one-factor-at-a-time (OAT), to see what effect this produces on the output.

OAT customarily involves
	- moving one parameter, keeping others at their baseline (nominal) values, then,
	- returning the parameter to its nominal value, then repeating for each of the other parameters in the same way.

Sensitivity may then be measured by monitoring changes in the output. This appears a logical approach as any change observed in the output will unambiguously be due to the single parameter changed. Furthermore, by changing one parameter at a time, one can keep all other parameters fixed to their central or baseline values. This increases the comparability of the results (all 'effects' are computed with reference to the same central point in space)


If we want to check the sensitivity of the HBV hydrological model performance to predict stream flow to each parameter, the One-At-a Time sensitivity analysis is agreat meathod that helps in this area 
OAT fixes the value of all parameters and change the value of one parameter within boundaries each time to check the result of the given function based on different values of one of the inputs

First of all to run the HBV lumped model which we need to test its 
performance (based on RMSE error) based on a defined range for each parameter 

Steps:
	* Run the model with the baseline parameter :ref:`1`
	* Define wrapper function and type :ref:`2`
	* Instantiate the SensitivityAnalysis object :ref:`3`
	* Run the OAT method :ref:`4`
	* Display the result with the SOBOL plot :ref:`5`

.. _1:

Run the model
--------------

.. code-block:: ipython3
	:linenos:
	
	import pandas as pd

	import Hapi.rrm.hbv_bergestrom92 as HBVLumped
	from Hapi.run import Run
	from Hapi.catchment import Catchment
	from Hapi.rrm.routing import Routing
	import Hapi.statistics.performancecriteria as PC
	from Hapi.statistics.sensitivityanalysis import SensitivityAnalysis as SA

	Parameterpath = "/data/Lumped/Coello_Lumped2021-03-08_muskingum.txt"

	Path = "/data/Lumped/"

	### meteorological data
	start = "2009-01-01"
	end = "2011-12-31"
	name = "Coello"
	Coello = Catchment(name, start, end)
	Coello.ReadLumpedInputs(Path + "meteo_data-MSWEP.csv")

	### Basic_inputs
	# catchment area
	CatArea = 1530
	# temporal resolution
	# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
	InitialCond = [0,10,10,10,0]

	Coello.ReadLumpedModel(HBVLumped, CatArea, InitialCond)

	### parameters
	 # no snow subroutine
	Snow = 0
	# if routing using Maxbas True, if Muskingum False
	Maxbas = False
	Coello.ReadParameters(Parameterpath, Snow, Maxbas=Maxbas)

	parameters = pd.read_csv(Parameterpath, index_col = 0, header = None)
	parameters.rename(columns={1:'value'}, inplace=True)

	UB = pd.read_csv(Path + "/UB-1-Muskinguk.txt", index_col = 0, header = None)
	parnames = UB.index
	UB = UB[1].tolist()
	LB = pd.read_csv(Path + "/LB-1-Muskinguk.txt", index_col = 0, header = None)
	LB = LB[1].tolist()
	Coello.ReadParametersBounds(UB, LB, Snow)

	# observed flow
	Coello.ReadDischargeGauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
	### Routing
	Route=1
	# RoutingFn=Routing.TriangularRouting2
	RoutingFn = Routing.Muskingum

	### run the model
	Run.RunLumped(Coello, Route, RoutingFn)

- Measure the performance of the baseline parameters

.. code:: ipython3

	Metrics = dict()
	Qobs = Coello.QGauges[Coello.QGauges.columns[0]]

	Metrics['RMSE'] = PC.RMSE(Qobs, Coello.Qsim['q'])
	Metrics['NSE'] = PC.NSE(Qobs, Coello.Qsim['q'])
	Metrics['NSEhf'] = PC.NSEHF(Qobs, Coello.Qsim['q'])
	Metrics['KGE'] = PC.KGE(Qobs, Coello.Qsim['q'])
	Metrics['WB'] = PC.WB(Qobs, Coello.Qsim['q'])

	print("RMSE= " + str(round(Metrics['RMSE'],2)))
	print("NSE= " + str(round(Metrics['NSE'],2)))
	print("NSEhf= " + str(round(Metrics['NSEhf'],2)))
	print("KGE= " + str(round(Metrics['KGE'],2)))
	print("WB= " + str(round(Metrics['WB'],2)))

.. _2:
Define wrapper function and type
----------------------------------------

Define the wrapper function to the OAT method and put the parameters argument
at the first position, and then list all the other arguments required for your function

the following defined function contains two inner function that calculates discharge for lumped HBV model and calculates the RMSE of the calculated discharge.

the first function `RUN.RunLumped` takes some arguments we need to pass it through the `OAT` method 
[ConceptualModel,data,p2,init_st,snow,Routing, RoutingFn] with the same order in the defined function "wrapper"

the second function is RMSE takes the calculated discharge from the first function and measured discharge array

to define the argument of the "wrapper" function
1- the random parameters valiable i=of the first function should be the first argument "wrapper(Randpar)"
2- the first function arguments with the same order (except that the parameter argument is taken out and placed at the first potition step-1)
3- list the argument of the second function with the same order that the second function takes them

There are two types of wrappers 
- The first one returns one value (performance metric)

.. code-block:: ipython3
	:linenos:

	# For Type 1
	def WrapperType1(Randpar,Route, RoutingFn, Qobs):
	    Coello.Parameters = Randpar

	    Run.RunLumped(Coello, Route, RoutingFn)
	    rmse = PC.RMSE(Qobs, Coello.Qsim['q'])
	    return rmse


.. _3:
Instantiate the SensitivityAnalysis object
-------------------------------------------

.. code-block:: ipython3
	:linenos:

	fn = WrapperType2

	Positions = [10]

	Sen = SA(parameters,Coello.LB, Coello.UB, fn, Positions, 5, Type=Type)

.. _4:
Run the OAT method
-------------------

.. code-block:: ipython3
	:linenos:
	Sen.OAT(Route, RoutingFn, Qobs)

.. _5:
Display the result with the SOBOL plot
---------------------------------------

.. code-block:: ipython3
	:linenos:

	From = ''
	To = ''
	
	    fig, ax1 = Sen.Sobol(RealValues=False, Title="Sensitivity Analysis of the RMSE to models parameters",
	              xlabel = "Maxbas Values", ylabel="RMSE", From=From, To=To,xlabel2='Time',
	              ylabel2='Discharge m3/s', spaces=[None,None,None,None,None,None])
	

The second type 
----------------

- The second wrapper returns two values (the performance metric and the calculated output from the model)

.. code-block:: ipython3
	:linenos:

	# For Type 2
	def WrapperType2(Randpar,Route, RoutingFn, Qobs):
	    Coello.Parameters = Randpar

	    Run.RunLumped(Coello, Route, RoutingFn)
	    rmse = PC.RMSE(Qobs, Coello.Qsim['q'])
	    return rmse, Coello.Qsim['q']


        fig, (ax1,ax2) = Sen.Sobol(RealValues=False, Title="Sensitivity Analysis of the RMSE to models parameters",
              xlabel = "Maxbas Values", ylabel="RMSE", From=From, To=To,xlabel2='Time',
              ylabel2='Discharge m3/s', spaces=[None,None,None,None,None,None])
	    From = 0
	    To = len(Qobs.values)
	    ax2.plot(Qobs.values[From:To], label='Observed', color='red')






First the SensitivityAnalysis method takes 4 arguments :

    1-parameters:previous obtained parameters
    2-LB: upper bound
    3-UB: lower bound
    4-wrapper: defined function contains the function you want to run with different parameters and the metric function you want to assess the first function based on it.



SensitivityAnalysis method returns a dictionary with the name of the parameters
as keys,
Each parameter has a disctionary with two keys 0: list of parameters woth relative values
1: list of parameter values

.. code:: ipython3

	import matplotlib.pyplot as plt
	import Hapi.performancecriteria as PC
	import Hapi.statisticaltools as ST
