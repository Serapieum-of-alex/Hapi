*****
Lumped Model Run
*****
To run the HBV lumped model inside Hapi you need to prepare the meteorological inputs (rainfall, temperature and potential evapotranspiration), HBV parameters, and the HBV model (you can load Bergström, 1992 version of HBV from Hapi )

- First load the prepared lumped version of the HBV module inside Hapi, the triangular routing function and the wrapper function that runs the lumped model `RUN`.

.. code-block:: py
    :linenos:

    import Hapi.rrm.hbv_bergestrom92 as HBVLumped
    from Hapi.run import Run
    from Hapi.catchment import Catchment
    from Hapi.rrm.routing import Routing

- read the meteorological data, data has be in the form of numpy array with the following order [rainfall, ET, Temp, Tm], ET is the potential evapotranspiration, Temp is the temperature (C), and Tm is the long term monthly average temperature.

.. code-block:: py
    :linenos:

    Parameterpath = Comp + "/data/lumped/Coello_Lumped2021-03-08_muskingum.txt"
    MeteoDataPath = Comp + "/data/lumped/meteo_data-MSWEP.csv"

    ### meteorological data
    start = "2009-01-01"
    end = "2011-12-31"
    name = "Coello"
    Coello = Catchment(name, start, end)
    Coello.readLumpedInputs(MeteoDataPath)

- Meteorological data

.. code-block:: py
    :linenos:

    start = "2009-01-01"
    end = "2011-12-31"
    name = "Coello"
    Coello = Catchment(name, start, end)
    Coello.readLumpedInputs(MeteoDataPath)

- Lumped model

    prepare the initial conditions, cathcment area and the lumped model.

.. code-block:: py
    :linenos:

    # catchment area
    AreaCoeff = 1530
    # [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
    InitialCond = [0,10,10,10,0]

    Coello.readLumpedModel(HBVLumped, AreaCoeff, InitialCond)

- Load the pre-estimated parameters
    snow option (if you want to simulate snow accumulation and snow melt or not)

.. code-block:: py
    :linenos:

    Snow = 0 # no snow subroutine
    # if routing using Maxbas True, if Muskingum False
    Coello.readParameters(Parameterpath, Snow)


- Prepare the routing options.

.. code-block:: py
    :linenos:

    # RoutingFn = Routing.TriangularRouting2
    RoutingFn = Routing.Muskingum_V
    Route = 1

- now all the data required for the model are prepared in the right form, now you can call the `runLumped` wrapper to initiate the calculation

.. code-block:: py
    :linenos:

    Run.runLumped(Coello, Route, RoutingFn)

to calculate some metrics for the quality assessment of the calculate discharge the `performancecriteria` contains some metrics like `RMSE`, `NSE`, `KGE` and `WB` , you need to load it, a measured time series of doscharge for the same period of the simulation is also needed for the comparison.

all methods in `performancecriteria` takes two numpy arrays of the same length and return real number.

.. code-block:: py
    :linenos:
    import Hapi.statistics.performancecriteria as PC

    Metrics = dict()
    Qobs = Coello.QGauges['q']

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

To plot the calculated and measured discharge import matplotlib

.. code-block:: py
    :linenos:

    gaugei = 0
    plotstart = "2009-01-01"
    plotend = "2011-12-31"
    Coello.plotHydrograph(plotstart, plotend, gaugei, Title= "Lumped Model")


    .. image:: /img/lumpedmodel.png
    :width: 400pt

- To save the results

.. code-block:: py
    :linenos:

    StartDate = "2009-01-01"
    EndDate = "2010-04-20"

    Path = SaveTo + "Results-Lumped-Model" + str(dt.datetime.now())[0:10] + ".txt"
    Coello.saveResults(Result=5, StartDate=StartDate, EndDate=EndDate, Path=Path)
