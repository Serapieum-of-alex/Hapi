# Distributed Hydrological Model

After preparing all the meteorological, GIS inputs required for the model, and Extracting the parameters for the catchment 

- Import the Catchment object which contains all the methods that reads the meteorological, GIS and 

	
	class Catchment():
    """
    ================================
        Catchment
    ================================
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
    """

    def __init__(self, name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                 TemporalResolution = "Daily"):
        """
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

        Returns
        -------
        None.

        """