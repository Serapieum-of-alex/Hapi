import datetime as dt
import numpy as np

# from Hapi.run import Run
import statista.metrics as PC
from Hapi.rrm.routing import Routing
from Hapi.rrm.calibration import Calibration
import Hapi.rrm.hbv_bergestrom92 as HBVLumped


def test_ReadParametersBounds(
        coello_rrm_date: list,
        lower_bound: list,
        upper_bound: list,
):
    Coello = Calibration('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Maxbas = True
    Snow = False
    Coello.ReadParametersBounds(lower_bound, upper_bound, Snow, Maxbas=Maxbas)
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
    Coello = Calibration('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadLumpedInputs(lumped_meteo_data_path)
    Coello.ReadLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Maxbas = True
    Coello.ReadParametersBounds(lower_bound, upper_bound, coello_Snow, Maxbas=Maxbas)

    parameters = []
    # Routing
    Route = 1
    RoutingFn = Routing.TriangularRouting1

    Basic_inputs = dict(Route=Route, RoutingFn=RoutingFn, InitialValues=parameters)

    # discharge gauges
    Coello.ReadDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)

    OF_args = []
    OF = PC.RMSE

    Coello.ReadObjectiveFn(OF, OF_args)

    ApiObjArgs = dict(hms=100, hmcr=0.95, par=0.65, dbw=2000, fileout=1, xinit=0,
                      filename=history_files)

    for i in range(len(ApiObjArgs)):
        print(list(ApiObjArgs.keys())[i], str(ApiObjArgs[list(ApiObjArgs.keys())[i]]))

    # pll_type = 'POA'
    pll_type = None

    ApiSolveArgs = dict(store_sol=True, display_opts=True, store_hst=False, hot_start=False)

    OptimizationArgs = [ApiObjArgs, pll_type, ApiSolveArgs]

    # cal_parameters = Coello.LumpedCalibration(Basic_inputs, OptimizationArgs, printError=None)

    # assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ['q']