from Hapi.hm.interface import Interface

def create_interface_instance(dates: list):
    Interface('Rhine', start=dates[0])

def test_ReadLateralsTable(
        dates: list,
        river_cross_section_path: str,
        interface_Laterals_table_path: str,
):
    IF = Interface('Rhine', start=dates[0])
    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)

    assert len(IF.LateralsTable) == 9 and len(IF.LateralsTable.columns) == 2

def test_ReadLaterals(
        dates: list,
        river_cross_section_path: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
):
    IF = Interface('Rhine', start=dates[0])
    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)
    assert len(IF.Laterals) == 80 and len(IF.Laterals.columns) == 10

def test_ReadBoundaryConditionsTable(
        dates: list,
        interface_bc_path: str,
):
    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)

    assert len(IF.BCTable) == 2 and len(IF.BCTable.columns) == 2

def test_ReadBoundaryConditions(
        dates: list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str
):
    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    assert len(IF.BC) == 80 and len(IF.BC.columns) == 3