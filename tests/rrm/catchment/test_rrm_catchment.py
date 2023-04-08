from types import ModuleType
from typing import Dict, List
import datetime as dt
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex

import Hapi.rrm.hbv_bergestrom92 as HBVLumped
from Hapi.catchment import Catchment
from Hapi.rrm.routing import Routing
from Hapi.run import Run
import Hapi.rrm.hbv_bergestrom92 as HBV


def test_create_catchment_instance(coello_rrm_date: list):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    assert Coello.dt == 1
    assert isinstance(Coello.Index, DatetimeIndex)
    assert isinstance(Coello.RouteRiver, str)


def test_read_lumped_meteo_inputs(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedInputs(lumped_meteo_data_path)
    assert isinstance(Coello.data, np.ndarray)


def test_read_lumped_model(
    coello_rrm_date: list,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    assert isinstance(Coello.LumpedModel, ModuleType)
    assert isinstance(Coello.CatArea, float)
    assert isinstance(Coello.InitialCond, list)


def test_read_lumped_ReadParameters(
    coello_rrm_date: list,
    lumped_parameters_path: str,
    coello_Snow: int,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readParameters(lumped_parameters_path, coello_Snow)
    assert isinstance(Coello.Parameters, list)
    assert Coello.Snow == coello_Snow


def test_ReadDischargeGauges(
    coello_rrm_date: list,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
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
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedInputs(lumped_meteo_data_path)
    Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Coello.readParameters(lumped_parameters_path, coello_Snow)
    # discharge gauges
    Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    RoutingFn = Routing.Muskingum_V
    Route = 1
    Run.runLumped(Coello, Route, RoutingFn)

    assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ["q"]


# TODO: still not finished as it does not run the plotHydrograph method
def test_PlotHydrograph(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_parameters_path: str,
    coello_Snow: int,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedInputs(lumped_meteo_data_path)
    Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Coello.readParameters(lumped_parameters_path, coello_Snow)
    # discharge gauges
    Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    RoutingFn = Routing.Muskingum_V
    Route = 1
    Run.runLumped(Coello, Route, RoutingFn)

    assert len(Coello.Qsim) == 1095 and Coello.Qsim.columns.to_list() == ["q"]


def test_save_lumped_results(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_parameters_path: str,
    coello_Snow: int,
    lumped_gauges_path: str,
    coello_gauges_date_fmt: str,
):
    Coello = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    Coello.readLumpedInputs(lumped_meteo_data_path)
    Coello.readLumpedModel(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    Coello.readParameters(lumped_parameters_path, coello_Snow)
    # discharge gauges
    Coello.readDischargeGauges(lumped_gauges_path, fmt=coello_gauges_date_fmt)
    RoutingFn = Routing.Muskingum_V
    Route = 1
    Run.runLumped(Coello, Route, RoutingFn)
    Path = "examples/hydrological-model/data/lumped_model/test-Lumped-Model_results.txt"
    Coello.saveResults(Result=5, path=Path)


class TestDistributed:
    def test_create_catchment_instance(
        self, coello_start_date: str, coello_end_date: str
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            SpatialResolution="Distributed",
            TemporalResolution="Daily",
            fmt="%Y-%m-%d",
        )
        assert coello.SpatialResolution == "distributed"
        assert coello.RouteRiver == "Muskingum"
        assert isinstance(coello.start, dt.datetime)

    def test_read_meteo_inputs(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_evap_path: str,
        coello_prec_path: str,
        coello_temp_path: str,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            SpatialResolution="Distributed",
            TemporalResolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.readRainfall(
            coello_evap_path, start=coello_start_date, end=coello_end_date
        )
        coello.readTemperature(
            coello_prec_path, start=coello_start_date, end=coello_end_date
        )
        coello.readET(coello_temp_path, start=coello_start_date, end=coello_end_date)
        assert isinstance(coello.Prec, np.ndarray)
        assert isinstance(coello.Temp, np.ndarray)
        assert isinstance(coello.ET, np.ndarray)
        assert coello.Prec.shape == (13, 14, 11)
        assert coello.ET.shape == (13, 14, 11)

    def test_read_gis_inputs(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_fd_path: str,
        coello_acc_path: str,
        coello_fdt: Dict,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            SpatialResolution="Distributed",
            TemporalResolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.readFlowAcc(coello_acc_path)
        coello.readFlowDir(coello_fd_path)
        assert coello.Outlet[0][0] == 10
        assert coello.Outlet[1][0] == 13
        assert coello.acc_val == [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            11,
            13,
            15,
            16,
            17,
            23,
            43,
            44,
            48,
            55,
            59,
            63,
            86,
            88,
        ]
        assert isinstance(coello.FlowDirArr, np.ndarray)
        assert coello.FlowDirArr.shape == (13, 14)
        assert coello.FDT == coello_fdt

    def test_read_lumped_model(
        self,
        coello_start_date: str,
        coello_end_date: str,
        coello_cat_area: int,
        coello_initial_cond: List,
    ):
        coello = Catchment(
            "coello",
            coello_start_date,
            coello_end_date,
            SpatialResolution="Distributed",
            TemporalResolution="Daily",
            fmt="%Y-%m-%d",
        )
        coello.readLumpedModel(HBV, coello_cat_area, coello_initial_cond)
        assert coello.LumpedModel == HBV
        assert coello.CatArea == coello_cat_area
        assert coello.InitialCond == coello_initial_cond
