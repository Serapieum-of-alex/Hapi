import Hapi.hm.calibration as RC


def test_create_calibration_instance(
        version: int,

):
    RC.Calibration("HM", version=version)


def test_ReadGaugesTable_method(calibration_gauges_table_path: str):
    Calib = RC.Calibration("HM", version=3)
    Calib.ReadGaugesTable(calibration_gauges_table_path)
    assert len(Calib.GaugesTable) == 3 and len(Calib.GaugesTable.columns) == 10


def test_ReadObservedQ(
        calibration_gauges_table_path: str,
        calibration_ReadObservedQ_Path: str,
        dates: list,
        nodatavalu: int,
        calibration_gauges_file_extension: str,
        gauge_date_format: str,
        version: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(calibration_gauges_table_path)
    Calib.ReadObservedQ(calibration_ReadObservedQ_Path, dates[0], dates[1],
                        nodatavalu, file_extension=calibration_gauges_file_extension,
                        gauge_date_format=gauge_date_format)

    assert len(Calib.QGauges) == 80 and len(Calib.QGauges.columns) == 3 and len(Calib.GaugesTable.columns) == 12


def test_ReadObservedWL(
        calibration_gauges_table_path: str,
        calibration_ReadObservedWL_Path: str,
        dates: list,
        nodatavalu: int,
        calibration_gauges_file_extension: str,
        gauge_date_format: str,
        version: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(calibration_gauges_table_path)
    Calib.ReadObservedWL(calibration_ReadObservedWL_Path, dates[0], dates[1],
                        nodatavalu, file_extension=calibration_gauges_file_extension,
                        gauge_date_format=gauge_date_format)
    assert len(Calib.WLGauges) == 80 and len(Calib.WLGauges.columns) == 3 and len(Calib.GaugesTable.columns) == 12
