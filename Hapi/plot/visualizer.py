""".Visualizer."""
import datetime as dt
import math
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleopatra.styles import Styles
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from scipy.stats import gumbel_r

hours = list(range(1, 25))


class Visualize:
    """Visualize.

    Visualize class contains different method to animate water surface profile
    spatial data change over time, styling methods for plotting

    Methods
    -------
        1- AnimateArray
        2- CrossSections
        3- WaterSurfaceProfile
        4- CrossSections
    """

    FigureDefaultOptions = dict(
        ylabel="",
        xlabel="",
        legend="",
        legend_size=10,
        fig_size=(10, 8),
        label_size=10,
        font_size=10,
        name="hist.tif",
        color1="#3D59AB",
        color2="#DC143C",
        line_width=3,
        axis_font_size=15,
    )

    def __init__(self, resolution: str = "Hourly"):
        self.resolution = resolution
        self.Anim = None

    @staticmethod
    def getLineStyle(style: Union[str, int] = "loosely dotted"):
        """LineStyle.

        Line styles for plotting

        Parameters
        ----------
        style : TYPE, optional
            DESCRIPTION. The default is 'loosely dotted'.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return Styles.get_line_style(style)

    @staticmethod
    def getMarkerStyle(style: int):
        """MarkerStyle.

        Marker styles for plotting

        Parameters
        ----------
        style : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        return Styles.get_marker_style(style)

    def plotGroundSurface(
        self,
        Sub,
        from_xs: Optional[int] = None,
        to_xs: Optional[int] = None,
        floodplain: bool = False,
        plot_lateral: bool = False,
        xlabels_number: int = 10,
        fig_size: tuple = (20, 10),
        laterals_color: Union[str, tuple] = "red",
        laterals_line_width: int = 1,
        option: int = 1,
        size: int = 50,
    ):
        """Plot the longitudinal profile of the segment.

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        from_xs : TYPE, optional
            DESCRIPTION. The default is ''.
        to_xs : TYPE, optional
            DESCRIPTION. The default is ''.
        floodplain : TYPE, optional
            DESCRIPTION. The default is False.
        plot_lateral : TYPE, optional
            DESCRIPTION. The default is False.
        xlabels_number: [int]
            Default is 10
        fig_size: [tuple]
            Default is (20, 10)
        laterals_color: [str, tuple]
            Defaut is "red",
        laterals_line_width: [int]
            Default is 1.
        option: [int]
            Default is 1
        size: [int]
        Default is 50.

        Returns
        -------
        None.
        """
        GroundSurfacefig = plt.figure(70, figsize=fig_size)
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=GroundSurfacefig)
        axGS = GroundSurfacefig.add_subplot(gs[0:2, 0:6])

        if not from_xs:
            from_xs = Sub.xs_names[0]

        if not to_xs:
            to_xs = Sub.xs_names[-1]

        # not the whole sub-basin
        axGS.set_xticks(list(range(from_xs, to_xs)))
        axGS.set_xticklabels(list(range(from_xs, to_xs)))

        axGS.set_xlim(from_xs - 1, to_xs + 1)

        axGS.tick_params(labelsize=8)
        # plot dikes
        axGS.plot(
            Sub.xs_names,
            Sub.cross_sections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        axGS.plot(
            Sub.xs_names,
            Sub.cross_sections["zr"],
            "k.-",
            linewidth=2,
            label="Right Dike",
        )

        if floodplain:
            fpl = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hl"]
            )
            fpr = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hr"]
            )
            axGS.plot(Sub.xs_names, fpl, "b-.", linewidth=2, label="Floodplain left")
            axGS.plot(Sub.xs_names, fpr, "r-.", linewidth=2, label="Floodplain right")

        if plot_lateral:
            if isinstance(Sub.laterals_table, list) and len(Sub.laterals_table) > 0:
                if option == 1:
                    # plot location of laterals
                    for i in range(len(Sub.laterals_table)):
                        axGS.vlines(
                            Sub.laterals_table[i],
                            0,
                            int(Sub.results_1d["q"].max()),
                            colors=laterals_color,
                            linestyles="dashed",
                            linewidth=laterals_line_width,
                        )
                else:
                    lat = pd.DataFrame()
                    lat["xsid"] = Sub.laterals_table
                    lat = lat.merge(Sub.cross_sections, on="xsid", how="left")

                    axGS.scatter(
                        Sub.laterals_table,
                        lat["gl"].tolist(),
                        c=laterals_color,
                        linewidth=laterals_line_width,
                        zorder=10,
                        s=size,
                    )
            else:
                print(" Please Read the Laterals data")

        maxelevel1 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zr"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        maxelevel2 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zl"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        maxlelv = max(maxelevel1, maxelevel2)
        minlev = Sub.cross_sections.loc[
            Sub.cross_sections["xsid"] == to_xs, "gl"
        ].values
        axGS.set_ylim(minlev - 5, maxlelv + 5)

        # plot the bedlevel/baklevel
        if Sub.version == 1:
            axGS.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            axGS.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            axGS.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"] + Sub.cross_sections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if xlabels_number != "":
            start, end = axGS.get_xlim()
            label_list = [int(i) for i in np.linspace(start, end, xlabels_number)]
            axGS.xaxis.set_ticks(label_list)

        title = "Water surface Profile Simulation Subid = {0}".format(Sub.id)
        axGS.set_title(title, fontsize=15)
        axGS.legend(fontsize=15)
        axGS.set_xlabel("Profile", fontsize=15)
        axGS.set_ylabel("Elevation m", fontsize=15)
        axGS.grid()
        GroundSurfacefig.tight_layout()

    def WaterSurfaceProfile(
        self,
        Sub,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        fps: int = 100,
        from_xs: Optional[int] = None,
        to_xs: Optional[int] = None,
        fmt: str = "%Y-%m-%d",
        fig_size: tuple = (20, 10),
        text_location: tuple = (1, 1),
        laterals_color: Union[int, str] = "#3D59AB",
        laterals_line_width: int = 1,
        x_axis_label_size: int = 10,
        y_axis_label_size: int = 10,
        xlabels_number: int = 10,
        x_tick_label_size: int = 8,
        last_river_reach: bool = True,
        floodplain: bool = True,
        repeat: bool = True,
    ) -> FuncAnimation:
        """WaterSurfaceProfile.

        Plot water surface profile

        Parameters
        ----------
        Sub : [Object]
            Reach-object created as a sub class from River object.
        start : [datetime object]
            starting date of the simulation.
        end : [datetime object]
            end date of the simulation.
        fps : [integer], optional
             It is an optional integer value that represents the delay between
             each frame in milliseconds. Its default is 100.
        from_xs : [integer], optional
            number of cross sections to be displayed before the chosen cross
            section . The default is 10.
        to_xs : [integer], optional
            number of cross sections to be displayed after the chosen cross
            section . The default is 10.
        x_tick_label_size: []

        xlabels_number:[]

        y_axis_label_size: []

        laterals_line_width: []

        x_axis_label_size:[]

        laterals_color: []

        text_location: []

        fmt: []

        fig_size: []

        last_river_reach: [bool]
            Default is True.
        floodplain: [bool]
            Default is True.
        repeat: [bool]
            Defaut is True
        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)

        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)
        msg = """The start date does not exist in the results, Results are
            between {0} and  {1}""".format(
            Sub.first_day, Sub.last_day
        )
        assert start in Sub.reference_index_results, msg

        msg = """ The end date does not exist in the results, Results are
            between {0} and  {1}""".format(
            Sub.first_day, Sub.last_day
        )
        assert end in Sub.reference_index_results, msg

        msg = """please read the boundary condition files using the
            'ReadBoundaryConditions' method """
        assert hasattr(Sub, "QBC"), msg

        msg = """ start Simulation date should be before the end simulation
            date """
        assert start < end, msg

        if Sub.from_beginning == 1:
            Period = Sub.days_list[
                np.where(Sub.reference_index == start)[0][0] : np.where(
                    Sub.reference_index == end
                )[0][0]
                + 1
            ]
        else:
            ii = Sub.date_to_ordinal(start)
            ii2 = Sub.date_to_ordinal(end)
            Period = list(range(ii, ii2 + 1))

        counter = [(i, j) for i in Period for j in hours]

        fig = plt.figure(60, figsize=fig_size)
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 2:6])
        ax1.set_ylim(0, int(Sub.results_1d["q"].max()))

        if not from_xs:
            # xs = 0
            # plot the whole sub-basin
            from_xs = Sub.xs_names[0]
        else:
            # xs = 1
            # not the whole sub-basin
            if from_xs < Sub.xs_names[0]:
                from_xs = Sub.xs_names[0]

        if not to_xs:
            to_xs = Sub.xs_names[-1]
        else:
            if to_xs > Sub.xs_names[-1]:
                to_xs = Sub.xs_names[-1]

        ax1.set_xlim(from_xs - 1, to_xs + 1)
        ax1.set_xticks(list(range(from_xs, to_xs + 1)))
        ax1.set_xticklabels(list(range(from_xs, to_xs + 1)))

        ax1.tick_params(labelsize=x_tick_label_size)
        ax1.locator_params(axis="x", nbins=xlabels_number)

        ax1.set_xlabel("Cross section No", fontsize=x_axis_label_size)
        ax1.set_ylabel("Discharge (m3/s)", fontsize=y_axis_label_size, labelpad=0.3)
        ax1.set_title("Reach-Basin" + " " + str(Sub.id), fontsize=15)
        ax1.legend(["Discharge"], fontsize=15)

        # plot location of laterals
        for i in range(len(Sub.laterals_table)):
            ax1.vlines(
                Sub.laterals_table[i],
                0,
                int(Sub.results_1d["q"].max()),
                colors=laterals_color,
                linestyles="dashed",
                linewidth=laterals_line_width,
            )

        lat = pd.DataFrame()
        lat["xsid"] = Sub.laterals_table
        lat = lat.merge(Sub.cross_sections, on="xsid", how="left")

        lim = ax1.get_ylim()
        y = np.ones(len(Sub.laterals_table), dtype=int) * (lim[1] - 50)
        lat = ax1.scatter(
            Sub.laterals_table,
            y,
            c=laterals_color,
            linewidth=laterals_line_width,
            zorder=10,
            s=50,
        )

        (q_line,) = ax1.plot([], [], linewidth=5)
        ax1.grid()

        ### BC
        # Q
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(1, 25)
        ax2.set_ylim(0, int(Sub.QBC.max().max()) + 1)

        ax2.set_xlabel("Time", fontsize=y_axis_label_size)
        ax2.set_ylabel("Q (m3/s)", fontsize=y_axis_label_size, labelpad=0.1)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=15)

        (bc_q_line,) = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=300)
        ax2.grid()

        # h
        ax3 = fig.add_subplot(gs[0, 0])
        ax3.set_xlim(1, 25)
        ax3.set_ylim(float(Sub.HBC.min().min()), float(Sub.HBC.max().max()))

        ax3.set_xlabel("Time", fontsize=y_axis_label_size)
        ax3.set_ylabel("water level", fontsize=y_axis_label_size, labelpad=0.5)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        (bc_h_line,) = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig.add_subplot(gs[1, 0:6])

        ymax1 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zr"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        ymax2 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zl"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        ymax = max(ymax1, ymax2)
        minlev = Sub.cross_sections.loc[
            Sub.cross_sections["xsid"] == to_xs, "gl"
        ].values
        ax4.set_ylim(minlev - 5, ymax + 5)
        ax4.set_xlim(from_xs - 1, to_xs + 1)
        ax4.set_xticks(list(range(from_xs, to_xs + 1)))
        ax4.set_xticklabels(list(range(from_xs, to_xs + 1)))

        ax4.tick_params(labelsize=x_tick_label_size)
        ax4.locator_params(axis="x", nbins=xlabels_number)

        ax4.plot(
            Sub.xs_names,
            Sub.cross_sections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        ax4.plot(
            Sub.xs_names,
            Sub.cross_sections["zr"],
            "k.-",
            linewidth=2,
            label="Right Dike",
        )

        if Sub.version == 1:
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"] + Sub.cross_sections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if floodplain:
            fpl = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hl"]
            )
            fpr = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hr"]
            )
            ax4.plot(Sub.xs_names, fpl, "b-.", linewidth=2, label="Floodplain left")
            ax4.plot(Sub.xs_names, fpr, "r-.", linewidth=2, label="Floodplain right")

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(fontsize=15)
        ax4.set_xlabel("Profile", fontsize=y_axis_label_size)
        ax4.set_ylabel("Elevation m", fontsize=y_axis_label_size)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.laterals_table)):
            ymin = Sub.cross_sections.loc[
                Sub.cross_sections["xsid"] == Sub.laterals_table[i], "gl"
            ].values[0]
            ax4.vlines(
                Sub.laterals_table[i],
                ymin,
                ymax,
                colors=laterals_color,
                linestyles="dashed",
                linewidth=laterals_line_width,
            )

        day_text = ax4.annotate(
            "",
            xy=(
                from_xs + text_location[0],
                Sub.cross_sections.loc[Sub.cross_sections["xsid"] == to_xs, "gl"].values
                + text_location[1],
            ),
            fontsize=20,
        )

        (wl_line,) = ax4.plot([], [], linewidth=5)
        (hLline,) = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()

        def init_q():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            hLline.set_data([], [])
            day_text.set_text("")

            bc_q_line.set_data([], [])
            bc_h_line.set_data([], [])
            # bc_q_point
            # bc_h_point
            lat.set_sizes([])

            return (
                q_line,
                wl_line,
                hLline,
                day_text,
                bc_q_line,
                bc_h_line,
                bc_q_point,
                bc_h_point,
                lat,
            )

        # animation function. this is called sequentially
        def animate_q(i):
            x = Sub.xs_names
            y = Sub.results_1d.loc[Sub.results_1d["day"] == counter[i][0], "q"][
                Sub.results_1d["hour"] == counter[i][1]
            ]
            # the Saintvenant subroutine writes the
            # results of the last xs in the next segment with the current
            # segment
            if not last_river_reach:
                y = y.values[:-1]

            q_line.set_data(x, y)

            day = Sub.reference_index.loc[counter[i][0], "date"]

            if len(Sub.laterals_table) > 0:
                lat.set_sizes(
                    sizes=Sub.Laterals.loc[day, Sub.laterals_table].values * 100
                )

            day_text.set_text("day = " + str(day + dt.timedelta(hours=counter[i][1])))

            y = Sub.results_1d.loc[Sub.results_1d["day"] == counter[i][0], "wl"][
                Sub.results_1d["hour"] == counter[i][1]
            ]
            # the Saintvenant subroutine writes the results
            # of the last xs in the next segment with the current segment
            if not last_river_reach:
                y = y.values[:-1]

            wl_line.set_data(x, y)

            y = (
                Sub.results_1d.loc[Sub.results_1d["day"] == counter[i][0], "h"][
                    Sub.results_1d["hour"] == counter[i][1]
                ]
                * 2
            )
            # temporary as now the Saintvenant subroutine writes the results
            # of the last xs in the next segment with the current segment
            if not last_river_reach:
                y = y.values[:-1]

            y = (
                y
                + Sub.cross_sections.loc[
                    Sub.cross_sections.index[len(Sub.xs_names) - 1], "gl"
                ]
            )
            hLline.set_data(x, y)

            x = Sub.QBC.columns.values

            y = Sub.QBC.loc[Sub.reference_index.loc[counter[i][0], "date"]].values
            bc_q_line.set_data(x, y)

            y = Sub.HBC.loc[Sub.reference_index.loc[counter[i][0], "date"]].values
            bc_h_line.set_data(x, y)

            x = counter[i][1]
            y = Sub.reference_index.loc[counter[i][0], "date"]
            scatter1 = ax2.scatter(x, Sub.QBC[x][y], s=300)
            scatter2 = ax3.scatter(x, Sub.HBC[x][y], s=300)

            return (
                q_line,
                wl_line,
                hLline,
                day_text,
                bc_q_line,
                bc_h_line,
                scatter1,
                scatter1,
                scatter2,
                lat,
            )

        Anim = FuncAnimation(
            fig,
            animate_q,
            init_func=init_q,
            frames=np.shape(counter)[0],
            interval=fps,
            blit=True,
            repeat=repeat,
        )
        self.Anim = Anim
        return Anim

    def WaterSurfaceProfile1Min(
        self,
        Sub,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        interval: float = 0.00002,
        from_xs: Union[str, int] = "",
        to_xs: Union[str, int] = "",
        fmt: str = "%Y-%m-%d",
        figsize: tuple = (20, 10),
        text_location: tuple = (1, 1),
        laterals_color: Union[str, tuple] = "#3D59AB",
        laterals_line_width: int = 1,
        x_axis_label_size: int = 10,
        y_axis_label_size: int = 10,
        xlabels_number: int = 20,
        x_tick_label_size: int = 8,
        floodplain: bool = True,
        repeat: bool = True,
    ) -> FuncAnimation:
        """WaterSurfaceProfile1Min.

        Plot water surface profile for 1 min data

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
        interval : TYPE, optional
            DESCRIPTION. The default is 0.00002.
        from_xs : TYPE, optional
            DESCRIPTION. The default is 10.
        to_xs : TYPE, optional
            DESCRIPTION. The default is 10.
        floodplain: [bool]
            Default is True.
        fmt: [str]
            Default is "%Y-%m-%d".
        figsize: [tuple]
            Default is (20, 10).
        text_location: [tuple]
            Default is (1, 1).
        laterals_color: [str]
            Default is "#3D59AB".
        laterals_line_width: [int]
            Default is 1.
        x_axis_label_size: [int]
            Default is 10.
        y_axis_label_size: [int]
            Default is 10.
        xlabels_number: [int]
            Default is 20.
        x_tick_label_size: [int]
            Default is 8.
        floodplain: bool
            Default is True.
        repeat: [bool]
            Defaut is True

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt) - dt.timedelta(minutes=int(Sub.dt / 60))

        assert start in Sub.h.index, (
            "plot start date in not in the 1min results, the results starts from "
            + str(Sub.h.index[0])
            + " - and ends on "
            + str(Sub.h.index[-1])
        )
        assert end in Sub.h.index, (
            "plot end date in not in the 1min results, the results starts from "
            + str(Sub.h.index[0])
            + " - and ends on "
            + str(Sub.h.index[-1])
        )

        counter = Sub.h.index[
            np.where(Sub.h.index == start)[0][0] : np.where(Sub.h.index == end)[0][0]
        ]
        nstep = (
            len(pd.date_range(start, start + dt.timedelta(days=1), freq=Sub.freq)) - 1
        )

        fig2 = plt.figure(20, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig2)

        ax1 = fig2.add_subplot(gs[0, 2:6])

        if from_xs == "":
            from_xs = Sub.xs_names[0]
            # to_xs = Reach.xs_names[-1]
        else:
            if from_xs < Sub.xs_names[0]:
                from_xs = Sub.xs_names[0]

            # if to_xs > Reach.xs_names[-1]:
            # to_xs = Reach.xs_names[-1]

        if to_xs == "":
            to_xs = Sub.xs_names[-1]
        else:
            if to_xs > Sub.xs_names[-1]:
                to_xs = Sub.xs_names[-1]

        ax1.set_xlim(from_xs - 1, to_xs + 1)

        ax1.set_xticks(list(range(from_xs, to_xs + 1)))
        ax1.set_xticklabels(list(range(from_xs, to_xs + 1)))

        ax1.tick_params(labelsize=x_tick_label_size)
        ax1.locator_params(axis="x", nbins=xlabels_number)

        ax1.set_xlabel("Cross section No", fontsize=x_axis_label_size)
        ax1.set_ylabel("Discharge (m3/s)", fontsize=y_axis_label_size, labelpad=0.5)
        ax1.set_title("Reach-Basin" + " " + str(Sub.id), fontsize=15)
        ax1.legend(["Discharge"], fontsize=15)
        ax1.set_ylim(0, int(Sub.q.max().max()))

        if Sub.version < 4:
            # ax1.set_ylim(0, int(Reach.results_1d['q'].max()))

            # plot location of laterals
            for i in range(len(Sub.laterals_table)):
                ax1.vlines(
                    Sub.laterals_table[i],
                    0,
                    int(int(Sub.q.max().max())),
                    colors=laterals_color,
                    linestyles="dashed",
                    linewidth=laterals_line_width,
                )

            lat = pd.DataFrame()
            lat["xsid"] = Sub.laterals_table
            lat = lat.merge(Sub.cross_sections, on="xsid", how="left")

            lim = ax1.get_ylim()
            y = np.ones(len(Sub.laterals_table), dtype=int) * (lim[1] - 50)
            lat = ax1.scatter(
                Sub.laterals_table,
                y,
                c=laterals_color,
                linewidth=laterals_line_width,
                zorder=10,
                s=50,
            )
        else:
            ax1.set_ylim(0, int(Sub.q.max().max()))

        (q_line,) = ax1.plot([], [], linewidth=5)
        ax1.grid()

        ### BC
        # Q
        ax2 = fig2.add_subplot(gs[0, 1:2])
        ax2.set_xlim(1, nstep)
        if Sub.version < 4:
            ax2.set_ylim(0, int(Sub.q_bc_1min.max().max()))
        else:
            ax2.set_ylim(0, int(Sub.USBC.max()))

        ax2.set_xlabel("Time", fontsize=y_axis_label_size)
        ax2.set_ylabel("Q (m3/s)", fontsize=y_axis_label_size, labelpad=0.1)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=15)

        (bc_q_line,) = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=150)
        ax2.grid()

        # h
        ax3 = fig2.add_subplot(gs[0, 0:1])
        ax3.set_xlim(1, nstep)
        if Sub.version < 4:
            ax3.set_ylim(
                float(Sub.h_bc_1min.min().min()), float(Sub.h_bc_1min.max().max())
            )

        ax3.set_xlabel("Time", fontsize=y_axis_label_size)
        ax3.set_ylabel("water level", fontsize=y_axis_label_size, labelpad=0.5)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        (bc_h_line,) = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=150)

        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        ymax1 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zr"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        ymax2 = max(
            Sub.cross_sections.loc[Sub.cross_sections["xsid"] >= from_xs, "zl"][
                Sub.cross_sections["xsid"] <= to_xs
            ]
        )
        ymax = max(ymax1, ymax2)
        minlev = Sub.cross_sections.loc[
            Sub.cross_sections["xsid"] == to_xs, "gl"
        ].values
        ax4.set_ylim(minlev - 5, ymax + 5)
        ax4.set_xlim(from_xs - 1, to_xs + 1)
        ax4.set_xticks(list(range(from_xs, to_xs + 1)))

        ax4.tick_params(labelsize=x_axis_label_size)
        ax4.locator_params(axis="x", nbins=xlabels_number)

        ax4.plot(
            Sub.xs_names,
            Sub.cross_sections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        ax4.plot(
            Sub.xs_names,
            Sub.cross_sections["zr"],
            "k.-",
            linewidth=2,
            label="Right Dike",
        )

        if Sub.version == 1:
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["gl"] + Sub.cross_sections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if floodplain:
            fpl = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hl"]
            )
            fpr = (
                Sub.cross_sections["gl"]
                + Sub.cross_sections["dbf"]
                + Sub.cross_sections["hr"]
            )
            ax4.plot(Sub.xs_names, fpl, "b-.", linewidth=2, label="Floodplain left")
            ax4.plot(Sub.xs_names, fpr, "r-.", linewidth=2, label="Floodplain right")

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(fontsize=10)
        ax4.set_xlabel("Profile", fontsize=10)
        ax4.set_ylabel("Elevation m", fontsize=10)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.laterals_table)):
            ymin = Sub.cross_sections.loc[
                Sub.cross_sections["xsid"] == Sub.laterals_table[i], "gl"
            ].values[0]
            ax4.vlines(
                Sub.laterals_table[i],
                ymin,
                ymax,
                colors=laterals_color,
                linestyles="dashed",
                linewidth=laterals_line_width,
            )

        day_text = ax4.annotate(
            "",
            xy=(
                from_xs + text_location[0],
                Sub.cross_sections.loc[Sub.cross_sections["xsid"] == to_xs, "gl"].values
                + text_location[1],
            ),
            fontsize=20,
        )

        (wl_line,) = ax4.plot([], [], linewidth=5)
        (hLline,) = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()

        # animation
        def init_min():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            day_text.set_text("")
            bc_q_line.set_data([], [])
            bc_h_line.set_data([], [])
            # bc_q_point
            # bc_h_point
            lat.set_sizes([])

            return (
                q_line,
                wl_line,
                bc_q_line,
                bc_h_line,
                bc_q_point,
                bc_h_point,
                day_text,
                lat,
            )

        # animation function. this is called sequentially
        def animate_min(i):
            day_text.set_text("Date = " + str(counter[i]))
            # discharge (ax1)
            x = Sub.xs_names
            y = Sub.q[Sub.q.index == counter[i]].values[0]
            q_line.set_data(x, y)

            # water level (ax4)
            y = Sub.h.loc[Sub.q.index == counter[i]].values[0]
            wl_line.set_data(x, y)

            day = counter[i].floor(freq="D")

            lat.set_sizes(sizes=Sub.Laterals.loc[day, Sub.laterals_table].values * 100)

            # BC Q (ax2)

            x = Sub.q_bc_1min.columns.values

            y = Sub.q_bc_1min.loc[day].values
            bc_q_line.set_data(x, y)

            # BC H (ax3)
            y = Sub.h_bc_1min.loc[day].values
            bc_h_line.set_data(x, y)

            # BC Q point (ax2)
            x = ((counter[i] - day).seconds / 60) + 1
            scatter1 = ax2.scatter(x, Sub.q_bc_1min[x][day], s=150)
            # BC h point (ax3)
            scatter2 = ax3.scatter(x, Sub.h_bc_1min[x][day], s=150)

            return (
                q_line,
                wl_line,
                bc_q_line,
                bc_h_line,
                scatter1,
                scatter2,
                day_text,
                lat,
            )

        anim = FuncAnimation(
            fig2,
            animate_min,
            init_func=init_min,
            frames=np.shape(counter)[0],
            interval=interval,
            blit=True,
            repeat=repeat,
        )
        self.Anim = anim
        return anim

    def river1d(
        self,
        Sub,
        start,
        end,
        interval=0.00002,
        xs=0,
        xs_before=10,
        xs_after=10,
        fmt="%Y-%m-%d",
        text_location=2,
        x_axis_label_size=15,
        y_axis_label_size=15,
        xlabels_number=50,
        plot_bankfull_depth=False,
    ):
        """river1d.

        plot river 1D

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
        interval : TYPE, optional
            DESCRIPTION. The default is 0.00002.
        xs : TYPE, optional
            DESCRIPTION. The default is 0.
        xs_before : TYPE, optional
            DESCRIPTION. The default is 10.
        xs_after : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)

        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        msg = (
            "plot start date in not in the 1min results, the results starts"
            " from {0} - and ends on {1} "
        )

        assert start in Sub.reference_index_results, msg.format(
            Sub.reference_index_results[0], Sub.reference_index_results[-1]
        )

        assert end in Sub.reference_index_results, msg.format(
            Sub.reference_index_results[0], Sub.reference_index_results[-1]
        )

        counter = Sub.reference_index_results[
            np.where(Sub.reference_index_results == start)[0][0] : np.where(
                Sub.reference_index_results == end
            )[0][0]
            + 1
        ]

        margin = 10
        fig2 = plt.figure(20, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig2)

        # USBC
        ax1 = fig2.add_subplot(gs[0, 0])
        # ax1.set_xlim(1, 1440)
        ax1.set_ylim(0, int(Sub.usbc.max() + margin))
        ax1.set_xlabel("Time", fontsize=15)
        if Sub.usbc.columns[0] == "q":
            # ax1.set_ylabel('USBC - Q (m3/s)', font_size=15)
            ax1.set_title("USBC - Q (m3/s)", fontsize=20)
        else:
            # ax1.set_ylabel('USBC - H (m)', font_size=15)
            ax1.set_title("USBC - H (m)", fontsize=20)
        # ax1.legend(["Q"], font_size=10)
        ax1.set_xlim(1, 25)
        (usbc_line,) = ax1.plot([], [], linewidth=5)
        # usbc_point = ax1.scatter([], [], s=150)
        ax1.grid()

        ax2 = fig2.add_subplot(gs[0, 1:5])
        if xs == 0:
            # plot the whole sub-basin
            ax2.set_xlim(Sub.xs_names[0] - 1, Sub.xs_names[-1] + 1)
            ax2.set_xticks(Sub.xs_names)
            ax2.set_xticklabels(Sub.xs_names)

            FigureFirstXS = Sub.xs_names[0]
            FigureLastXS = Sub.xs_names[-1]
        else:
            # not the whole sub-basin
            FigureFirstXS = Sub.xs_names[xs] - xs_before
            if FigureFirstXS < Sub.xs_names[0]:
                FigureFirstXS = Sub.xs_names[0]

            FigureLastXS = Sub.xs_names[xs] + xs_after
            if FigureLastXS > Sub.xs_names[-1]:
                FigureLastXS = Sub.xs_names[-1]

            ax2.set_xlim(FigureFirstXS, FigureLastXS)
            ax2.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax2.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        ax2.set_ylim(np.nanmin(Sub.q) - 10, int(np.nanmax(Sub.q)) + 10)
        ax2.tick_params(labelsize=x_axis_label_size)
        ax2.locator_params(axis="x", nbins=xlabels_number)
        ax2.set_xlabel("Cross section No", fontsize=x_axis_label_size)
        ax2.set_title("Discharge (m3/s)", fontsize=20)
        # ax2.set_ylabel('Discharge (m3/s)', font_size=y_axis_label_size, labelpad=0.5)
        ax2.legend(["Discharge"], fontsize=15)

        (q_line,) = ax2.plot([], [], linewidth=5)
        ax2.grid()

        ### BC

        # DSBC
        ax3 = fig2.add_subplot(gs[0, 5:6])
        ax3.set_xlim(1, 25)
        ax3.set_ylim(0, float(Sub.dsbc.min() + margin))

        ax3.set_xlabel("Time", fontsize=15)
        if Sub.dsbc.columns[0] == "q":
            # ax3.set_ylabel('DSBC', font_size=15, labelpad=0.5)
            ax3.set_title("DSBC - Q (m3/s)", fontsize=20)
        else:
            # ax3.set_ylabel('USBC', font_size=15, labelpad=0.5)
            ax3.set_title("DSBC - H(m)", fontsize=20)

        # ax3.legend(["WL"], font_size=10)

        (dsbc_line,) = ax3.plot([], [], linewidth=5)
        # dsbc_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        if xs == 0:
            ax4.set_xlim(Sub.xs_names[0] - 1, Sub.xs_names[-1] + 1)
            ax4.set_xticks(Sub.xs_names)
            ymin = Sub.cross_sections.loc[
                Sub.cross_sections["xsid"] == FigureFirstXS, "bed level"
            ].values.min()
            ymax = Sub.cross_sections.loc[
                Sub.cross_sections["xsid"] == FigureFirstXS, "bed level"
            ].values.max()
            ax4.set_ylim(ymin, ymax + np.nanmax(Sub.h) + 5)
        else:
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(
                Sub.cross_sections.loc[
                    Sub.cross_sections["xsid"] == FigureFirstXS, "bed level"
                ].values,
                Sub.cross_sections.loc[
                    Sub.cross_sections["xsid"] == FigureLastXS, "zr"
                ].values
                + 5,
            )

        ax4.tick_params(labelsize=x_axis_label_size)
        ax4.locator_params(axis="x", nbins=xlabels_number)

        ax4.plot(
            Sub.xs_names,
            Sub.cross_sections["bed level"],
            "k-",
            linewidth=5,
            label="Ground level",
        )
        if plot_bankfull_depth:
            ax4.plot(
                Sub.xs_names,
                Sub.cross_sections["bed level"] + Sub.cross_sections["depth"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(["Ground level", "Bankful depth"], fontsize=10)
        ax4.set_xlabel("Profile", fontsize=10)
        ax4.set_ylabel("Elevation m", fontsize=10)
        ax4.grid()

        if xs == 0:
            text_location = text_location + Sub.xs_names[0]
            day_text = ax4.annotate(
                " ",
                xy=(text_location, Sub.cross_sections["bed level"].min() + 1),
                fontsize=20,
            )
        else:
            day_text = ax4.annotate(
                " ",
                xy=(
                    FigureFirstXS + text_location,
                    Sub.cross_sections.loc[
                        Sub.cross_sections["xsid"] == FigureLastXS, "gl"
                    ].values
                    + 1,
                ),
                fontsize=20,
            )

        (wl_line,) = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()

        # animation
        def init_min():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            day_text.set_text("")
            usbc_line.set_data([], [])
            dsbc_line.set_data([], [])
            # bc_q_point
            return q_line, wl_line, day_text, usbc_line, dsbc_line

        # animation function. this is called sequentially
        def animate_min(i):
            day_text.set_text("Date = " + str(counter[i]))

            # discharge (ax1)
            x = Sub.xs_names

            y = Sub.q[np.where(Sub.reference_index_results == counter[i])[0][0], :]
            q_line.set_data(x, y)

            # water level (ax4)
            y = Sub.wl[np.where(Sub.reference_index_results == counter[i])[0][0], :]
            wl_line.set_data(x, y)

            # USBC
            f = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            y = (
                Sub.usbc.loc[f : f + dt.timedelta(days=1)]
                .resample("H")
                .mean()
                .interpolate("linear")
                .values
            )
            x = hours[: len(y)]
            usbc_line.set_data(x, y)

            # DSBC
            y = (
                Sub.dsbc.loc[f : f + dt.timedelta(days=1)]
                .resample("H")
                .mean()
                .interpolate("linear")
                .values
            )
            dsbc_line.set_data(x, y)

            # x = counter[i][1]
            # y = Reach.reference_index.loc[counter[i][0], 'date']
            # ax2.scatter(x, Reach.QBC[x][y])

            # # BC Q point (ax2)
            # x = ((counter[i] - dt.datetime(counter[i].year, counter[i].month, counter[i].day)).seconds / 60) + 1
            # y = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            # ax2.scatter(x, Reach.USBC[x][y])

            return (
                q_line,
                wl_line,
                day_text,
                usbc_line,
                dsbc_line,
            )  # ax2.scatter(x, Reach.USBC[x][y], s=150),

        # plt.tight_layout()

        anim = FuncAnimation(
            fig2,
            animate_min,
            init_func=init_min,
            frames=len(counter),
            interval=interval,
            blit=True,
        )

        return anim

    def plotCrossSections(
        self,
        Sub,
        from_xs: Optional[int] = None,
        to_xs: Optional[int] = None,
        xs_rows: int = 3,
        xs_columns: int = 3,
        bedlevel: bool = False,
        title_size: int = 15,
        text_size: int = 15,
        fig_size: tuple = (18, 10),
        line_width: int = 6,
        same_scale: bool = False,
        text_spacing: List[tuple] = [(1, 1), (1, 2)],
        plotting_option: int = 1,
        plot_annotation: bool = True,
    ):
        """CrossSections.

        Plot cross sections of a river segment.

        Parameters
        ----------
        Sub : [Object]
            Reach-object created as a sub class from River object..
        from_xs : TYPE, optional
            DESCRIPTION. The default is ''.
        to_xs : TYPE, optional
            DESCRIPTION. The default is ''.
        xs_rows : TYPE, optional
            DESCRIPTION. The default is 3.
        xs_columns : TYPE, optional
            DESCRIPTION. The default is 3.
        bedlevel : TYPE, optional
            DESCRIPTION. The default is False.
        title_size : TYPE, optional
            DESCRIPTION. The default is 15.
        text_size : TYPE, optional
            DESCRIPTION. The default is 15.
        fig_size : TYPE, optional
            DESCRIPTION. The default is (18, 10).
        line_width : TYPE, optional
            DESCRIPTION. The default is 6.
        plotting_option : [integer]
            1 if you want to plot the whole cross-section, 2 to execlude the
            dikes(river bankfull area and floodplain will be plotted),
            3 to plot only the bankfull area.
        same_scale: [bool]
            Default is False.
        text_spacing: [tuple]
            Default is [(1, 1), (1, 2)].
        plot_annotation: [bool]
            Default is True.

        Returns
        -------
        None.
        """
        if not from_xs:
            startxs_ind = 0
        else:
            startxs_ind = Sub.xs_names.index(from_xs)

        if not to_xs:
            endxs_ind = Sub.xsno - 1
        else:
            endxs_ind = Sub.xs_names.index(to_xs)

        names = list(range(1, 17))
        XSS = pd.DataFrame(
            columns=names, index=Sub.cross_sections.loc[startxs_ind:endxs_ind, "xsid"]
        )

        # calculate the vertices of the cross sections
        for i in range(startxs_ind, endxs_ind + 1):
            ind = XSS.index[i - startxs_ind]
            ind2 = Sub.cross_sections.index[i]

            XSS[1].loc[XSS.index == ind] = 0
            XSS[2].loc[XSS.index == ind] = 0

            bl = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "bl"].values[
                0
            ]
            b = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "b"].values[0]
            br = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "br"].values[
                0
            ]

            XSS[3].loc[XSS.index == ind] = bl
            XSS[4].loc[XSS.index == ind] = bl
            XSS[5].loc[XSS.index == ind] = bl + b
            XSS[6].loc[XSS.index == ind] = bl + b
            XSS[7].loc[XSS.index == ind] = bl + b + br
            XSS[8].loc[XSS.index == ind] = bl + b + br

            gl = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "gl"].values[
                0
            ]

            if bedlevel:
                subtract = 0
            else:
                subtract = gl

            zl = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "zl"].values[
                0
            ]
            zr = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "zr"].values[
                0
            ]

            if "dbf" in Sub.cross_sections.columns:
                dbf = Sub.cross_sections.loc[
                    Sub.cross_sections.index == ind2, "dbf"
                ].values[0]

            hl = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "hl"].values[
                0
            ]
            hr = Sub.cross_sections.loc[Sub.cross_sections.index == ind2, "hr"].values[
                0
            ]

            XSS[9].loc[XSS.index == ind] = zl - subtract

            if "dbf" not in Sub.cross_sections.columns:
                XSS[10].loc[XSS.index == ind] = gl + hl - subtract
                XSS[11].loc[XSS.index == ind] = gl - subtract
                XSS[14].loc[XSS.index == ind] = gl - subtract
                XSS[15].loc[XSS.index == ind] = gl + hr - subtract
            else:
                XSS[10].loc[XSS.index == ind] = gl + dbf + hl - subtract
                XSS[11].loc[XSS.index == ind] = gl + dbf - subtract
                XSS[14].loc[XSS.index == ind] = gl + dbf - subtract
                XSS[15].loc[XSS.index == ind] = gl + dbf + hr - subtract

            XSS[12].loc[XSS.index == ind] = gl - subtract
            XSS[13].loc[XSS.index == ind] = gl - subtract

            XSS[16].loc[XSS.index == ind] = zr - subtract

        # plot the cross sections
        xsplot = len(range(startxs_ind, endxs_ind + 1))
        figno = int(math.ceil(xsplot / (xs_columns * xs_rows)))

        ind2 = startxs_ind
        ind = XSS.index[ind2 - startxs_ind]
        for i in range(figno):
            if same_scale:
                sharex = True
                sharey = True
            else:
                sharex = False
                sharey = False

            fig, ax_XS = plt.subplots(
                ncols=xs_columns,
                nrows=xs_rows,
                figsize=fig_size,
                sharex=sharex,
                sharey=sharey,
            )
            # gs = gridspec.GridSpec(xs_rows, xs_columns)

            for j in range(xs_rows):
                for k in range(xs_columns):
                    if ind2 <= endxs_ind:
                        XsId = Sub.cross_sections.loc[
                            Sub.cross_sections.index[ind2], "xsid"
                        ]
                        xcoord = (
                            XSS[names[0:8]].loc[XSS.index == ind].values.tolist()[0]
                        )
                        ycoord = (
                            XSS[names[8:16]].loc[XSS.index == ind].values.tolist()[0]
                        )
                        b = Sub.cross_sections.loc[
                            Sub.cross_sections["xsid"] == ind, "b"
                        ].values[0]
                        bl = Sub.cross_sections.loc[
                            Sub.cross_sections["xsid"] == ind, "bl"
                        ].values[0]
                        gl = Sub.cross_sections.loc[
                            Sub.cross_sections["xsid"] == ind, "gl"
                        ].values[0]

                        # ax_XS = fig.add_subplot(gs[j, k])
                        if plotting_option == 1:
                            ax_XS[j, k].plot(xcoord, ycoord, linewidth=line_width)
                            x = text_spacing[0][0]
                            x1 = text_spacing[1][0]
                        elif plotting_option == 2:
                            ax_XS[j, k].plot(
                                xcoord[1:-1], ycoord[1:-1], linewidth=line_width
                            )
                            x = text_spacing[0][0] + bl
                            x1 = text_spacing[1][0] + bl
                        else:
                            ax_XS[j, k].plot(
                                xcoord[2:-2], ycoord[2:-2], linewidth=line_width
                            )
                            x = text_spacing[0][0] + bl
                            x1 = text_spacing[1][0] + bl

                        ax_XS[j, k].title.set_text("xs ID = " + str(XsId))
                        ax_XS[j, k].title.set_fontsize(title_size)

                        if same_scale:
                            # when sharex and sharey are true the labels
                            # disappear so set thier visability to true
                            ax_XS[j, k].xaxis.set_tick_params(labelbottom=True)
                            ax_XS[j, k].yaxis.set_tick_params(labelbottom=True)

                        if plot_annotation:
                            if Sub.version > 1:
                                dbf = Sub.cross_sections.loc[
                                    Sub.cross_sections["xsid"] == ind, "dbf"
                                ].values[0]
                                b = Sub.cross_sections.loc[
                                    Sub.cross_sections["xsid"] == ind, "b"
                                ].values[0]

                                if bedlevel:
                                    ax_XS[j, k].annotate(
                                        "dbf=" + str(round(dbf, 2)),
                                        xy=(x, gl + text_spacing[0][1]),
                                        fontsize=text_size,
                                    )

                                    ax_XS[j, k].annotate(
                                        "b=" + str(round(b, 2)),
                                        xy=(x1, gl + text_spacing[1][1]),
                                        fontsize=text_size,
                                    )
                                else:
                                    ax_XS[j, k].annotate(
                                        "dbf=" + str(round(dbf, 2)),
                                        xy=(x, text_spacing[0][1]),
                                        fontsize=text_size,
                                    )

                                    ax_XS[j, k].annotate(
                                        "b=" + str(round(b, 2)),
                                        xy=(x1, text_spacing[1][1]),
                                        fontsize=text_size,
                                    )

                        ind2 = ind2 + 1
                        ind = ind + 1

            plt.subplots_adjust(
                wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96
            )

        return fig, ax_XS

    def plot1minProfile(
        self,
        Sub,
        date: str,
        x_axis_label_size: int = 10,
        xlabels_number: int = 50,
        fmt: str = "%Y-%m-%d",
    ):
        """Plot1minProfile.

        Plot water surface profile for 1 min data.

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        date : TYPE
            DESCRIPTION.
        x_axis_label_size : TYPE, optional
            DESCRIPTION. The default is 10.
        xlabels_number : TYPE, optional
            DESCRIPTION. The default is 50.
        fmt : TYPE, optional
            DESCRIPTION. The default is "%Y-%m-%d".

        Returns
        -------
        None.
        """
        if isinstance(date, str):
            date = dt.datetime.strptime(date, fmt)

        fig50, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax1.plot(Sub.q.columns, Sub.q[Sub.q.index == date].values[0], "r")
        ax2.plot(Sub.h.columns, Sub.h[Sub.q.index == date].values[0])
        ax1.set_ylabel("Discharge", fontsize=20)
        ax2.set_ylabel("Water level", fontsize=20)
        ax1.set_xlabel("Cross sections", fontsize=20)
        ax1.set_xticks(Sub.xs_names)
        ax1.tick_params(labelsize=x_axis_label_size)
        ax1.locator_params(axis="x", nbins=xlabels_number)

        ax1.grid()

    @staticmethod
    def Histogram(v1, v2, NoAxis=2, filter1=0.2, Save=False, pdf=True, **kwargs):
        """Histogram.

        Histogram method plots the histogram of two given list of values

        Parameters
        ----------
            1-v1 : [List]
                first list of values.
            2-v2 : [List]
                second list of values.

        Returns
        -------
            - histogram plot.

        Example
        -------
        Vis.Histogram(Val1, val2,2,fig_size=(5.5,4.5), color1='#27408B',
                        xlabel = 'Inundation Depth (m)', ylabel = 'Frequency', legend_size = 15,
                         font_size=15, label_size = 15, axis_font_size = 11,
                         legend = ['RIM1.0', 'RIM2.0'], pdf = False, Save = False,
                         name = str(Event1.event_index.loc[end_ind,'id']))
        """
        # update the default options
        Fkeys = list(kwargs.keys())
        for key in Fkeys:
            if key in Visualize.FigureDefaultOptions.keys():
                Visualize.FigureDefaultOptions[key] = kwargs[key]

        v1 = np.array([j for j in v1 if j > filter1])
        v2 = np.array([j for j in v2 if j > filter1])

        if pdf:
            param_dist1 = gumbel_r.fit(np.array(v1))
            param_dist2 = gumbel_r.fit(np.array(v2))

            d1 = np.linspace(v1.min(), v1.max(), v1.size)
            d2 = np.linspace(v2.min(), v2.max(), v2.size)
            pdf_fitted1 = gumbel_r.pdf(d1, loc=param_dist1[0], scale=param_dist1[1])
            pdf_fitted2 = gumbel_r.pdf(d2, loc=param_dist2[0], scale=param_dist2[1])

        if NoAxis == 1:
            # if bins in kwargs.keys():
            plt.figure(60, figsize=(10, 8))
            n, bins, patches = plt.hist([v1, v2], color=["#3D59AB", "#DC143C"])
            # for key in kwargs.keys():
            #     if key == 'legend':
            #         plt.legend(kwargs['legend'])
            #     if key == 'legend size':
            #         plt.legend(kwargs['legend'],font_size = int(kwargs['legend_size']))
            #     if key == 'xlabel':
            #         plt.xlabel(kwargs['xlabel'])
            #     if key == 'ylabel':
            #         plt.ylabel(kwargs['ylabel'])
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']

        elif NoAxis == 2:
            fig, ax1 = plt.subplots(figsize=Visualize.FigureDefaultOptions["fig_size"])
            # n1= ax1.hist([v1,v2], bins=15, alpha = 0.7, color=[color1,color2])
            n1 = ax1.hist(
                [v1, v2],
                bins=15,
                alpha=0.7,
                color=[
                    Visualize.FigureDefaultOptions["color1"],
                    Visualize.FigureDefaultOptions["color2"],
                ],
            )
            # label=['RIM1.0','RIM2.0']) #width = 0.2,

            ax1.set_ylabel(
                "Frequency", fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"]
            )
            # ax1.yaxis.label.set_color(color1)
            ax1.set_xlabel(
                "Inundation Depth Ranges (m)",
                fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
            )

            # ax1.tick_params(axis='y', color = color1)
            # ax1.spines['right'].set_color(color1)
            if pdf:
                ax2 = ax1.twinx()
                ax2.plot(
                    d1,
                    pdf_fitted1,
                    "-.",
                    color=Visualize.FigureDefaultOptions["color1"],
                    linewidth=Visualize.FigureDefaultOptions["line_width"],
                    label="RIM1.0 pdf",
                )
                ax2.plot(
                    d2,
                    pdf_fitted2,
                    "-.",
                    color=Visualize.FigureDefaultOptions["color2"],
                    linewidth=Visualize.FigureDefaultOptions["line_width"],
                    label="RIM2.0 pdf",
                )
                ax2.set_ylabel(
                    "Probability density function (pdf)",
                    fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
                )
            # else:
            #     ax2.yaxis.set_ticklabels([])
            #     # ax2.yaxis.set_major_formatter(plt.NullFormatter())
            #     # ax2.tick_params(right='off', labelright='off')
            #     ax2.set_xticks([])
            #     ax2.tick_params(axis='y', color = color2)

            #     # if key == 'xlabel':
            #         # xlabel = kwargs['xlabel']
            #     # if key == 'xlabel':
            #         # xlabel = kwargs['xlabel']

            # n2 = ax2.hist(v2,  bins=n1[1], alpha = 0.4, color=color2)#width=0.2,
            # ax2.set_ylabel("Frequency", font_size = 15)
            # ax2.yaxis.label.set_color(color2)

            # ax2.tick_params(axis='y', color = color2)
            # plt.title("Reach-Basin = " + str(Subid), font_size = 15)

            # minall = min(min(n1[1]), min(n2[1]))
            # if minall < 0:
            #     minall =0

            # maxall = max(max(n1[1]), max(n2[1]))
            # ax1.set_xlim(minall, maxall)
            #    ax1.set_yticklabels(ax1.get_yticklabels(), color = color1)
            #    ax2.set_yticklabels(ax2.get_yticklabels(), color = color2)

            # # options
            # for key in kwargs.keys():
            # if key == 'legend':
            # ax1.legend(self.FigureOptions['legend'])
            # if key == 'legend_size':
            ax1.legend(
                kwargs["legend"],
                fontsize=int(Visualize.FigureDefaultOptions["legend_size"]),
            )
            # if key == 'xlabel':
            # ax1.set_xlabel(self.FigureOptions['xlabel'])
            # if key == 'ylabel':
            # ax1.set_ylabel(self.FigureOptions['ylabel'])
            # if key == 'label_size':
            ax1.set_xlabel(
                Visualize.FigureDefaultOptions["xlabel"],
                fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
            )
            ax1.set_ylabel(
                Visualize.FigureDefaultOptions["ylabel"],
                fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
            )
            # if key == 'font_size':
            plt.rcParams.update(
                {"font.size": int(Visualize.FigureDefaultOptions["axis_font_size"])}
            )

            # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,font_size = 15)

            plt.tight_layout()

        if Save:
            plt.savefig(
                Visualize.FigureDefaultOptions["name"] + "-hist.tif", transparent=True
            )
            # plt.close()

    def ListAttributes(self):
        """ListAttributes.

        Print Attributes List
        """
        print("\n")
        print(
            f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(f"{key} : {repr(self.__dict__[key])}")

        print("\n")
