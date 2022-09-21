from Hapi.hm.interface import Interface


def create_interface_instance(dates: list):
    Interface("Rhine", start=dates[0])


def test_ReadLateralsTable(
    dates: list,
    river_cross_section_path: str,
    interface_Laterals_table_path: str,
):
    IF = Interface("Rhine", start=dates[0])
    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)

    assert len(IF.LateralsTable) == 9 and len(IF.LateralsTable.columns) == 2


def test_ReadLaterals(
    dates: list,
    river_cross_section_path: str,
    interface_Laterals_table_path: str,
    interface_Laterals_folder: str,
    interface_Laterals_date_format: str,
    test_time_series_length: int,
):
    IF = Interface("Rhine", start=dates[0])
    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(
        path=interface_Laterals_folder, date_format=interface_Laterals_date_format
    )
    assert (
        len(IF.Laterals) == test_time_series_length and len(IF.Laterals.columns) == 10
    )


def test_ReadBoundaryConditionsTable(
    dates: list,
    interface_bc_path: str,
):
    IF = Interface("Rhine", start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)

    assert len(IF.BCTable) == 2 and len(IF.BCTable.columns) == 2


def test_ReadBoundaryConditions(
    dates: list,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    test_time_series_length: int,
):
    IF = Interface("Rhine", start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    assert len(IF.BC) == test_time_series_length and len(IF.BC.columns) == 3


def test_ReadRRMProgression(
    dates: list,
    river_cross_section_path: str,
    interface_Laterals_table_path: str,
    rrm_resutls_hm_location: str,
    interface_Laterals_date_format: str,
    laterals_number_ts: int,
    no_laterals: int,
):
    IF = Interface("Rhine", start=dates[0])
    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadRRMProgression(
        path=rrm_resutls_hm_location, date_format=interface_Laterals_date_format
    )
    assert len(IF.RRMProgression) == laterals_number_ts
    assert len(IF.RRMProgression.columns) == no_laterals
