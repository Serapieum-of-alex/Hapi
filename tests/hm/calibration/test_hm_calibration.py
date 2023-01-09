from typing import List

from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

import Hapi.hm.calibration as RC


def test_create_calibration_instance(
    version: int,
):
    RC.Calibration("HM", version=version)


def test_ReadGaugesTable(gauges_table_path: str):
    Calib = RC.Calibration("HM", version=3)
    Calib.readGaugesTable(gauges_table_path)
    assert len(Calib.hm_gauges) == 3 and len(Calib.hm_gauges.columns) == 10


def test_ReadObservedQ(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    version: int,
    test_time_series_length: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    assert (
        len(Calib.q_gauges) == test_time_series_length
        and len(Calib.q_gauges.columns) == 3
        and len(Calib.hm_gauges.columns) == 12
    )


def test_ReadObservedWL(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    version: int,
    test_time_series_length: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    assert (
        len(Calib.WLGauges) == test_time_series_length
        and len(Calib.WLGauges.columns) == 3
        and len(Calib.hm_gauges.columns) == 12
    )


def test_CalculateProfile(
    version: int,
    segment3: int,
    river_cross_section_path: str,
    river_network_path: str,
    calibrateProfile_DS_bedlevel: float,
    calibrateProfile_mn: float,
    calibrateProfile_slope: float,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.calculateProfile(
        segment3,
        calibrateProfile_DS_bedlevel,
        calibrateProfile_mn,
        calibrateProfile_slope,
    )

    assert (
        Calib.crosssections.loc[Calib.crosssections["id"] == 3, "gl"].tolist()[-1]
        == calibrateProfile_DS_bedlevel
    )


def test_SmoothMaxSlope(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.smoothMaxSlope(segment3)


def test_SmoothBedLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.smoothBedLevel(segment3)


def test_DownWardBedLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
    DownWardBedLevel_height: float,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.downWardBedLevel(segment3, DownWardBedLevel_height)


def test_SmoothBankLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.smoothBankLevel(segment3)


def test_SmoothFloodplainHeight(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.smoothFloodplainHeight(segment3)


def test_SmoothBedWidth(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.smoothBedWidth(segment3)


def test_CheckFloodplain(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.readXS(river_cross_section_path)
    Calib.checkFloodplain()


def test_ReadRRM(
    gauges_table_path: str,
    rrmpath: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readRRM(rrmpath, fmt="'%Y-%m-%d'")
    assert len(Calib.q_rrm) == test_time_series_length and len(
        Calib.q_rrm.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.q_rrm.columns.to_list() for elem in rrmgauges)


def test_ReadHMQ(
    gauges_table_path: str,
    hm_separated_q_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.q_hm) == test_time_series_length and len(
        Calib.q_hm.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.q_hm.columns.to_list() for elem in rrmgauges)


def test_ReadHMWL(
    gauges_table_path: str,
    hm_separated_wl_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readHMWL(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.WLHM) == test_time_series_length and len(
        Calib.WLHM.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.WLHM.columns.to_list() for elem in rrmgauges)


class Test_GetAnnualMax:
    def test_option_1(
        self,
        gauges_table_path: str,
        ObservedQ_long_ts_Path: str,
        ObservedQ_long_ts_dates: list,
        nodatavalu: int,
        gauge_long_ts_date_format: str,
        ObservedQ_long_ts_len: int,
        gauges_numbers: int,
    ):
        """test extracting max annual observed discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.readGaugesTable(gauges_table_path)
        Calib.readObservedQ(
            ObservedQ_long_ts_Path,
            ObservedQ_long_ts_dates[0],
            ObservedQ_long_ts_dates[1],
            nodatavalu,
            gauge_date_format=gauge_long_ts_date_format,
        )
        # RIM.ReadObservedQ(ObservedPath, GRDCStartDate, GRDCEndDate, NoValue)

        Calib.getAnnualMax(option=1)
        assert len(Calib.annual_max_obs_q) == ObservedQ_long_ts_len
        assert len(Calib.annual_max_obs_q.columns) == gauges_numbers

    def test_option_3(
        self,
        gauges_table_path: str,
        rrm_long_ts_number: int,
        gauges_numbers: int,
        rrmpath_long_ts: str,
    ):
        """test extracting max annual hydrologic model simulated discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.readGaugesTable(gauges_table_path)
        Calib.readRRM(rrmpath_long_ts, fmt="'%Y-%m-%d'")

        Calib.getAnnualMax(option=3)
        assert len(Calib.annual_max_rrm) == rrm_long_ts_number
        assert len(Calib.annual_max_rrm.columns) == gauges_numbers

    def test_option_4(
        self,
        gauges_table_path: str,
        hm_long_ts_number: int,
        gauges_numbers: int,
        hm_separated_results_q_long_ts_path,
    ):
        """test extracting max annual hydraulic model simulated discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.readGaugesTable(gauges_table_path)
        Calib.readHMQ(hm_separated_results_q_long_ts_path, fmt="'%Y-%m-%d'")

        Calib.getAnnualMax(option=4)
        assert len(Calib.annual_max_hm_q) == hm_long_ts_number
        assert len(Calib.annual_max_hm_q.columns) == gauges_numbers


def test_HMvsRRM(
    gauges_table_path: str,
    rrmpath: str,
    hm_separated_q_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.readRRM(rrmpath, fmt="'%Y-%m-%d'")
    Calib.HMvsRRM()
    assert isinstance(Calib.MetricsHMvsRRM, DataFrame) and isinstance(
        Calib.MetricsHMvsRRM, GeoDataFrame
    )
    assert len(Calib.MetricsHMvsRRM) == 3
    assert all(
        Calib.MetricsHMvsRRM.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(Calib.MetricsHMvsRRM.columns == Metrics_table_columns)


def test_RRMvsObserved(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    rrmpath: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readRRM(rrmpath, fmt="'%Y-%m-%d'")
    Calib.RRMvsObserved()
    assert isinstance(Calib.MetricsRRMvsObs, DataFrame) and isinstance(
        Calib.MetricsRRMvsObs, GeoDataFrame
    )
    assert len(Calib.MetricsRRMvsObs) == 3
    assert all(
        Calib.MetricsRRMvsObs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(elem in Calib.MetricsRRMvsObs.columns for elem in Metrics_table_columns)


def test_HMQvsObserved(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.HMQvsObserved()
    assert isinstance(Calib.MetricsHMQvsObs, DataFrame) and isinstance(
        Calib.MetricsHMQvsObs, GeoDataFrame
    )
    assert len(Calib.MetricsHMQvsObs) == 3
    assert all(
        Calib.MetricsHMQvsObs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(elem in Calib.MetricsHMQvsObs.columns for elem in Metrics_table_columns)


def test_HMWLvsObserved(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMWL(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.HMWLvsObserved()
    assert isinstance(Calib.MetricsHMWLvsObs, DataFrame) and isinstance(
        Calib.MetricsHMWLvsObs, GeoDataFrame
    )
    assert len(Calib.MetricsHMWLvsObs) == 3
    assert all(
        Calib.MetricsHMWLvsObs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(elem in Calib.MetricsHMWLvsObs.columns for elem in Metrics_table_columns)


def test_InspectGauge(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    ReadObservedQ_Path: str,
    rrmpath: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
    InspectGauge_sub_id: int,
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMWL(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.HMWLvsObserved()
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.HMQvsObserved()

    Calib.readRRM(rrmpath, fmt="'%Y-%m-%d'")
    Calib.HMvsRRM()
    Calib.RRMvsObserved()
    gaugei = 0
    summary, fig, ax = Calib.InspectGauge(InspectGauge_sub_id, gaugei=gaugei)
    assert isinstance(fig, Figure)
    assert isinstance(summary, DataFrame)
    assert all(elem in summary.columns.to_list() for elem in Metrics_table_columns)


def test_SaveMetices(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    ReadObservedQ_Path: str,
    rrmpath: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
    hm_saveto: str,
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMWL(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.HMWLvsObserved()
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.readHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.HMQvsObserved()

    Calib.readRRM(rrmpath, fmt="'%Y-%m-%d'")
    Calib.HMvsRRM()
    Calib.RRMvsObserved()
    Calib.SaveMetices(hm_saveto)
