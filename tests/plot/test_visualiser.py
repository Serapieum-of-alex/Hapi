import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import Hapi.hm.river as R
from Hapi.hm.interface import Interface
from Hapi.plot.visualizer import Visualize as V


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
    River.read_xs(river_cross_section_path)
    River.read_river_network(river_network_path)
    River.one_d_result_path = Read1DResult_path
    Sub = R.Reach(segment3, River)

    IF = Interface("Rhine", start=dates[0])
    IF.readBoundaryConditionsTable(interface_bc_path)
    IF.read_boundary_conditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    IF.read_xs(river_cross_section_path)
    IF.readLateralsTable(interface_Laterals_table_path)
    IF.readLaterals(
        path=interface_Laterals_folder, date_format=interface_Laterals_date_format
    )

    Sub.get_flow(IF)
    Sub.read_1d_results()

    Vis = V(resolution="Hourly")
    Vis.plotGroundSurface(
        Sub, floodplain=True, plot_lateral=True, xlabels_number=20, option=2
    )
    plt.close()
    Vis.plotGroundSurface(
        Sub, floodplain=True, plot_lateral=True, xlabels_number=20, option=1
    )
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
    River.read_xs(river_cross_section_path)
    River.read_river_network(river_network_path)
    River.one_d_result_path = Read1DResult_path
    Sub = R.Reach(segment3, River)
    Vis = V(resolution="Hourly")

    fig, ax = Vis.plotCrossSections(
        Sub,
        bedlevel=True,
        from_xs=plot_xs_seg3_fromxs,
        to_xs=plot_xs_seg3_toxs,
        same_scale=True,
        text_spacing=[(1, 1), (1, 4)],
        plotting_option=3,
    )
    plt.close()
    assert isinstance(fig, Figure)


# TODO: figure out how to close the animation after it finishes
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
#     River.one_d_result_path = Read1DResult_path
#     River.us_bc_path = usbc_path
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
#     Anim = Vis.WaterSurfaceProfile(Reach, animate_start, animate_end, fps=2, xlabels_number=5,
#                                    x_axis_label_size=10, text_location=(-1, -2),repeat=False)
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
#     River.one_d_result_path = Read1DResult_path
#     River.one_min_result_path = subdailyresults_path
#     River.us_bc_path = usbc_path
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
#     Reach.ReadSubDailyResults(animate_start, animate_end, last_river_reach=lastsegment)
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
    River.read_xs(river_cross_section_path)
    River.read_river_network(river_network_path)
    River.us_bc_path = usbc_path
    River.one_min_result_path = subdailyresults_path
    Sub = R.Reach(segment3, River)
    Sub.read_sub_daily_results(
        onemin_results_dates[0], onemin_results_dates[1], last_river_reach=lastsegment
    )

    Vis = V(resolution="Hourly")
    Vis.plot1minProfile(Sub, dates[0], xlabels_number=20)
