from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import Hapi.hm.river as R
from Hapi.hm.interface import Interface
from Hapi.plot.visualizer import Visualize as V


def test_create_visualize_instance():
    Vis = V(resolution="Hourly")
    assert isinstance(Vis.MarkerStyleList, list)
    assert isinstance(Vis.FigureDefaultOptions, dict)
    assert isinstance(Vis.linestyles, OrderedDict)


def test_GroundSurface(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    segment3: int,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    interface_Laterals_table_path: str,
    interface_Laterals_folder: str,
    interface_Laterals_date_format: str,
    Read1DResult_path: str,
):
    River = R.River("HM", version=version, start=dates[0])
    River.readXS(river_cross_section_path)
    River.readRiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path
    Sub = R.Reach(segment3, River)

    IF = Interface("Rhine", start=dates[0])
    IF.readBoundaryConditionsTable(interface_bc_path)
    IF.readBoundaryConditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    IF.readXS(river_cross_section_path)
    IF.readLateralsTable(interface_Laterals_table_path)
    IF.readLaterals(
        path=interface_Laterals_folder, date_format=interface_Laterals_date_format
    )

    Sub.getFlow(IF)
    Sub.read1DResult()

    Vis = V(resolution="Hourly")
    Vis.GroundSurface(Sub, floodplain=True, plotlateral=True, nxlabels=20, option=2)
    plt.close()
    Vis.GroundSurface(Sub, floodplain=True, plotlateral=True, nxlabels=20, option=1)
    plt.close()


def test_CrossSections(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    segment3: int,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    interface_Laterals_table_path: str,
    interface_Laterals_folder: str,
    interface_Laterals_date_format: str,
    Read1DResult_path: str,
    plot_xs_seg3_fromxs: int,
    plot_xs_seg3_toxs: int,
):
    River = R.River("HM", version=version, start=dates[0])
    River.readXS(river_cross_section_path)
    River.readRiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path
    Sub = R.Reach(segment3, River)
    Vis = V(resolution="Hourly")

    fig, ax = Vis.CrossSections(
        Sub,
        bedlevel=True,
        fromxs=plot_xs_seg3_fromxs,
        toxs=plot_xs_seg3_toxs,
        samescale=True,
        textspacing=[(1, 1), (1, 4)],
        plottingoption=3,
    )
    plt.close()
    assert isinstance(fig, Figure)


# TODO: figure put how to close the animation after it finishes
# def test_WaterSurfaceProfile(
#         version: int,
#         river_cross_section_path: str,
#         river_network_path: str,
#         dates:list,
#         segment3: int,
#         interface_bc_path: str,
#         interface_bc_folder: str,
#         interface_bc_date_format: str,
#         interface_Laterals_table_path: str,
#         interface_Laterals_folder: str,
#         interface_Laterals_date_format: str,
#         Read1DResult_path: str,
#         usbc_path: str,
#         animate_start: str,
#         animate_end: str
#
# ):
#     River = R.River('HM', version=version, start=dates[0])
#     River.ReadCrossSections(river_cross_section_path)
#     River.RiverNetwork(river_network_path)
#     River.onedresultpath = Read1DResult_path
#     River.usbcpath = usbc_path
#     Reach = R.Reach(segment3, River)
#
#     IF = Interface('Rhine', start=dates[0])
#     IF.ReadBoundaryConditionsTable(interface_bc_path)
#     IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)
#
#     IF.ReadCrossSections(river_cross_section_path)
#     IF.ReadLateralsTable(interface_Laterals_table_path)
#     IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)
#
#     Reach.GetFlow(IF)
#     Reach.Read1DResult()
#     Reach.ReadBoundaryConditions(start=dates[0], end=dates[1])
#
#     Vis = V(resolution="Hourly")
#     Anim = Vis.WaterSurfaceProfile(Reach, animate_start, animate_end, fps=2, nxlabels=5,
#                                    xaxislabelsize=10, textlocation=(-1, -2),repeat=False)
#     # rc('animation', html='jshtml')
#     plt.close()
#     assert isinstance(Anim, FuncAnimation)
#
#
# def test_WaterSurfaceProfile1Min(
#         version: int,
#         river_cross_section_path: str,
#         river_network_path: str,
#         dates:list,
#         segment3: int,
#         interface_bc_path: str,
#         interface_bc_folder: str,
#         interface_bc_date_format: str,
#         interface_Laterals_table_path: str,
#         interface_Laterals_folder: str,
#         interface_Laterals_date_format: str,
#         Read1DResult_path: str,
#         usbc_path: str,
#         animate_start: str,
#         animate_end: str,
#         lastsegment: bool,
#         subdailyresults_path: str,
#
# ):
#     River = R.River('HM', version=version, start=dates[0])
#     River.ReadCrossSections(river_cross_section_path)
#     River.RiverNetwork(river_network_path)
#     River.onedresultpath = Read1DResult_path
#     River.oneminresultpath = subdailyresults_path
#     River.usbcpath = usbc_path
#     Reach = R.Reach(segment3, River)
#
#     IF = Interface('Rhine', start=dates[0])
#     IF.ReadBoundaryConditionsTable(interface_bc_path)
#     IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)
#
#     IF.ReadCrossSections(river_cross_section_path)
#     IF.ReadLateralsTable(interface_Laterals_table_path)
#     IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)
#
#     Reach.GetFlow(IF)
#     Reach.Read1DResult()
#     Reach.ReadSubDailyResults(animate_start, animate_end, Lastsegment=lastsegment)
#     Reach.ReadBoundaryConditions(start=animate_start, end=animate_end)
#
#     Vis = V(resolution="Hourly")
#     Anim = Vis.WaterSurfaceProfile1Min(Reach, animate_start, animate_end,
#                                        interval=0.000000000000000000000000000000000001,
#                                        repeat=False)
#
#     plt.close()
#     assert isinstance(Anim, FuncAnimation)


def test_Plot1minProfile(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    segment3: int,
    subdailyresults_path: str,
    usbc_path: str,
    onemin_results_dates: list,
    lastsegment: bool,
):

    River = R.River("HM", version=version, start=dates[0])
    River.readXS(river_cross_section_path)
    River.readRiverNetwork(river_network_path)
    River.usbcpath = usbc_path
    River.oneminresultpath = subdailyresults_path
    Sub = R.Reach(segment3, River)
    Sub.readSubDailyResults(
        onemin_results_dates[0], onemin_results_dates[1], Lastsegment=lastsegment
    )

    Vis = V(resolution="Hourly")
    Vis.Plot1minProfile(Sub, dates[0], nxlabels=20)
