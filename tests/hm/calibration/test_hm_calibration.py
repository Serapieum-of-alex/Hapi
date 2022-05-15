from typing import List

import Hapi.hm.calibration as RC


def test_create_calibration_instance(
        version: int,
):
    RC.Calibration("HM", version=version)


def test_ReadGaugesTable(gauges_table_path: str):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(gauges_table_path)
    assert len(Calib.GaugesTable) == 3 and len(Calib.GaugesTable.columns) == 10


def test_ReadObservedQ(
        gauges_table_path: str,
        ReadObservedQ_Path: str,
        dates: list,
        nodatavalu: int,
        gauge_date_format: str,
        version: int,
        test_time_series_length: int
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedQ(ReadObservedQ_Path, dates[0], dates[1],
                        nodatavalu, gauge_date_format=gauge_date_format)

    assert len(Calib.QGauges) == test_time_series_length \
           and len(Calib.QGauges.columns) == 3 and \
           len(Calib.GaugesTable.columns) == 12


def test_ReadObservedWL(
        gauges_table_path: str,
        ReadObservedWL_Path: str,
        dates: list,
        nodatavalu: int,
        gauge_date_format: str,
        version: int,
        test_time_series_length: int
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedWL(ReadObservedWL_Path, dates[0], dates[1],
                         nodatavalu, gauge_date_format=gauge_date_format)
    assert len(Calib.WLGauges) == test_time_series_length and \
           len(Calib.WLGauges.columns) == 3 and \
           len(Calib.GaugesTable.columns) == 12


def test_CalculateProfile(
        version: int,
        segment3: int,
        river_cross_section_path: str,
        river_network_path: str,
        calibrateProfile_DS_bedlevel: float,
        calibrateProfile_mn: float,
        calibrateProfile_slope: float
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.CalculateProfile(segment3, calibrateProfile_DS_bedlevel, calibrateProfile_mn, calibrateProfile_slope)

    assert Calib.crosssections.loc[Calib.crosssections['id'] == 3, 'gl'].tolist()[-1] == calibrateProfile_DS_bedlevel


def test_SmoothMaxSlope(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.SmoothMaxSlope(segment3)


def test_SmoothBedLevel(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.SmoothBedLevel(segment3)


def test_DownWardBedLevel(
        version: int,
        river_cross_section_path: str,
        segment3: int,
        DownWardBedLevel_height: float
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.DownWardBedLevel(segment3, DownWardBedLevel_height)


def test_SmoothBankLevel(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.SmoothBankLevel(segment3)


def test_SmoothFloodplainHeight(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.SmoothFloodplainHeight(segment3)


def test_SmoothBedWidth(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.SmoothBedWidth(segment3)


def test_CheckFloodplain(
        version: int,
        river_cross_section_path: str,
        segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadCrossSections(river_cross_section_path)
    Calib.CheckFloodplain()


def test_ReadRRM(
        gauges_table_path: str,
        rrmpath: str,
        test_time_series_length: int,
        rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadRRM(rrmpath, fmt="'%Y-%m-%d'")
    assert len(Calib.QRRM) == test_time_series_length and len(Calib.QRRM.columns) == len(rrmgauges)
    assert all(elem in Calib.QRRM.columns.to_list() for elem in rrmgauges)


def test_ReadHMQ(
        gauges_table_path: str,
        hm_separated_q_results_path,
        test_time_series_length: int,
        rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadHMQ(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.QHM) == test_time_series_length and len(Calib.QHM.columns) == len(rrmgauges)
    assert all(elem in Calib.QHM.columns.to_list() for elem in rrmgauges)


def test_ReadHMWL(
        gauges_table_path: str,
        hm_separated_wl_results_path,
        test_time_series_length: int,
        rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadHMWL(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.WLHM) == test_time_series_length and len(Calib.WLHM.columns) == len(rrmgauges)
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
            gauges_numbers: int
    ):
        """
        test extracting max annual observed discharge
        """
        Calib = RC.Calibration("HM", version=3)
        Calib.ReadGaugesTable(gauges_table_path)
        Calib.ReadObservedQ(ObservedQ_long_ts_Path, ObservedQ_long_ts_dates[0], ObservedQ_long_ts_dates[1],
                            nodatavalu, gauge_date_format=gauge_long_ts_date_format)
        # RIM.ReadObservedQ(ObservedPath, GRDCStartDate, GRDCEndDate, NoValue)

        Calib.GetAnnualMax(option=1)
        assert len(Calib.AnnualMaxObsQ) == ObservedQ_long_ts_len
        assert len(Calib.AnnualMaxObsQ.columns) == gauges_numbers

    def test_option_3(
            self,
            gauges_table_path: str,
            rrm_long_ts_number: int,
            gauges_numbers: int,
            rrmpath_long_ts: str,
    ):
        """
        test extracting max annual hydrologic model simulated discharge
        """
        Calib = RC.Calibration("HM", version=3)
        Calib.ReadGaugesTable(gauges_table_path)
        Calib.ReadRRM(rrmpath_long_ts, fmt="'%Y-%m-%d'")

        Calib.GetAnnualMax(option=3)
        assert len(Calib.AnnualMaxRRM) == rrm_long_ts_number
        assert len(Calib.AnnualMaxRRM.columns) == gauges_numbers


    def test_option_4(
            self,
            gauges_table_path: str,
            hm_long_ts_number: int,
            gauges_numbers: int,
            hm_separated_results_q_long_ts_path,
    ):
        """
        test extracting max annual hydraulic model simulated discharge
        """
        Calib = RC.Calibration("HM", version=3)
        Calib.ReadGaugesTable(gauges_table_path)
        Calib.ReadHMQ(hm_separated_results_q_long_ts_path, fmt="'%Y-%m-%d'")

        Calib.GetAnnualMax(option=4)
        assert len(Calib.AnnualMaxHMQ) == hm_long_ts_number
        assert len(Calib.AnnualMaxHMQ.columns) == gauges_numbers


# def test_InspectGauge():
#     subid = 28
#     gaugei = 0
#     # start ="1990-01-01"
#     start = ""
#     end = ''  # "1994-3-1"
#
#     summary, fig, ax = Calib.InspectGauge(subid, gaugei=gaugei, start=start, end=end)