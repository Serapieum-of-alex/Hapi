import datetime as dt

import numpy as np

# from Hapi.run import Run
import statista.metrics as PC

import Hapi.rrm.hbv_bergestrom92 as HBVLumped
from Hapi.rrm.calibration import Calibration
from Hapi.rrm.routing import Routing


def test_ReadParametersBounds(
    coello_rrm_date: list,
    lower_bound: list,
    upper_bound: list,
):
    Coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Maxbas = True
    Snow = False
    Coello.readParametersBounds(lower_bound, upper_bound, Snow, Maxbas=Maxbas)
    assert isinstance(Coello.UB, np.ndarray)
    assert isinstance(Coello.LB, np.ndarray)
    assert isinstance(Coello.Snow, bool)
    assert isinstance(Coello.Maxbas, bool)


def test_LumpedCalibration(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_parameters_path: str,
    coello_Snow: bool,
    lower_bound: list,
    upper_bound: list,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
    history_files: str,
):
    Coello = Calibration("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedInputs(lumped_meteo_data_path)
    Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Maxbas = True
    Coello.readParametersBounds(lower_bound, upper_bound, coello_Snow, Maxbas=Maxbas)

    parameters = []
    # Routing
    Route = 1
    RoutingFn = Routing.TriangularRouting1

    Basic_inputs = dict(Route=Route, RoutingFn=RoutingFn, InitialValues=parameters)

    # discharge gauges
    Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)

    OF_args = []
    OF = PC.RMSE

    Coello.readObjectiveFn(OF, OF_args)

    ApiObjArgs = dict(
        hms=100,
        hmcr=0.95,
        par=0.65,
        dbw=2000,
        fileout=1,
        xinit=0,
        filename=history_files,
    )

    for i in range(len(ApiObjArgs)):
        print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

    # pll_type = 'POA'
    pll_type = None

    ApiSolveArgs = dict(
        store_sol=True, display_opts=True, store_hst=False, hot_start=False
    )

    OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]

    # cal_parameters = Coello.lumpedCalibration(Basic_inputs, OptimizationArgs, printError=None)

    # assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ['q']

class TestDistributed:
    def test_create_calibration_instance(
        self,
        coello_start_date: str,
        coello_end_date: str
        ):
        coello = Calibration(
            "coello",
            coello_start_date,
            coello_end_date,
            SpatialResolution = "Distributed",
            TemporalResolution = "Daily",
            fmt = "%Y-%m-%d"
        )
        assert coello.SpatialResolution == "Distributed"
        assert coello.RouteRiver == "Muskingum"
        assert isinstance(coello.start, dt.datetime)

