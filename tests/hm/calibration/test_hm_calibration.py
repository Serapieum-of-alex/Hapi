import Hapi.hm.calibration as RC


def test_create_calibration_instance(
        version: int,
):
    RC.Calibration("HM", version=version)


def test_ReadGaugesTable_method(gauges_table_path: str):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(gauges_table_path)
    assert len(Calib.GaugesTable) == 3 and len(Calib.GaugesTable.columns) == 10


def test_ReadObservedQ(
        gauges_table_path: str,
        ReadObservedQ_Path: str,
        dates: list,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        version: int,
        test_time_series_length: int
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedQ(ReadObservedQ_Path, dates[0], dates[1],
                        nodatavalu, file_extension=gauges_file_extension,
                        gauge_date_format=gauge_date_format)

    assert len(Calib.QGauges) == test_time_series_length \
           and len(Calib.QGauges.columns) == 3 and \
           len(Calib.GaugesTable.columns) == 12


def test_ReadObservedWL(
        gauges_table_path: str,
        ReadObservedWL_Path: str,
        dates: list,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        version: int,
        test_time_series_length: int
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedWL(ReadObservedWL_Path, dates[0], dates[1],
                         nodatavalu, file_extension=gauges_file_extension,
                         gauge_date_format=gauge_date_format)
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
