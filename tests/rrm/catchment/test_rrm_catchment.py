from pandas.core.indexes.datetimes import DatetimeIndex
import numpy as np
from pandas.core.frame import DataFrame
from types import ModuleType

from Hapi.catchment import Catchment
import Hapi.rrm.hbv_bergestrom92 as HBVLumped
from Hapi.rrm.routing import Routing
from Hapi.run import Run

def test_create_catchment_instance(
        coello_rrm_date: list
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    assert Coello.dt == 1
    assert isinstance(Coello.Index, DatetimeIndex)
    assert isinstance(Coello.RouteRiver, str)


def test_read_lumped_meteo_inputs(
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadLumpedInputs(lumped_meteo_data_path)
    assert isinstance(Coello.data, np.ndarray)


def test_read_lumped_model(
        coello_rrm_date: list,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    assert isinstance(Coello.LumpedModel, ModuleType)
    assert isinstance(Coello.CatArea, float)
    assert isinstance(Coello.InitialCond, list)


def test_read_lumped_ReadParameters(
        coello_rrm_date: list,
        lumped_parameters_path: str,
        coello_Snow: int,
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadParameters(lumped_parameters_path, coello_Snow)
    assert isinstance(Coello.Parameters, list)
    assert Coello.Snow == coello_Snow


def test_ReadDischargeGauges(
        coello_rrm_date: list,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    assert isinstance(Coello.QGauges, DataFrame)


def test_RunLumped(
        coello_rrm_date: list,
        lumped_meteo_data_path: str,
        coello_AreaCoeff: float,
        coello_InitialCond: list,
        lumped_parameters_path: str,
        coello_Snow: int,
        lumped_gauges_path: str,
        coello_gauges_date_fmt: str,
):
    Coello = Catchment('rrm', coello_rrm_date[0], coello_rrm_date[1])
    Coello.ReadLumpedInputs(lumped_meteo_data_path)
    Coello.ReadLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Coello.ReadParameters(lumped_parameters_path, coello_Snow)
    # discharge gauges
    Coello.ReadDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    RoutingFn = Routing.Muskingum_V
    Route = 1
    Run.RunLumped(Coello, Route, RoutingFn)

    assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ['q']