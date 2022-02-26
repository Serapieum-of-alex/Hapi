""".

Created on Sat Mar 14 16:36:01 2020

@author: mofarrag
"""
import datetime as dt
import math
import os
from collections import OrderedDict
from typing import List, Union

try :
    from osgeo import gdal
except ImportError:
    import gdal

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, gridspec
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
        figsize=(10, 8),
        labelsize=10,
        fontsize=10,
        name="hist.tif",
        color1="#3D59AB",
        color2="#DC143C",
        linewidth=3,
        Axisfontsize=15,
    )

    linestyles = OrderedDict(
        [
            ("solid", (0, ())),  # 0
            ("loosely dotted", (0, (1, 10))),  # 1
            ("dotted", (0, (1, 5))),  # 2
            ("densely dotted", (0, (1, 1))),  # 3
            ("loosely dashed", (0, (5, 10))),  # 4
            ("dashed", (0, (5, 5))),  # 5
            ("densely dashed", (0, (5, 1))),  # 6
            ("loosely dashdotted", (0, (3, 10, 1, 10))),  # 7
            ("dashdotted", (0, (3, 5, 1, 5))),  # 8
            ("densely dashdotted", (0, (3, 1, 1, 1))),  # 9
            ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),  # 10
            ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),  # 11
            ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),  # 12
            ("densely dashdotdottededited", (0, (6, 1, 1, 1, 1, 1))),# 13
        ]
    )

    MarkerStyleList = [
        "--o",
        ":D",
        "-.H",
        "--x",
        ":v",
        "--|",
        "-+",
        "-^",
        "--s",
        "-.*",
        "-.h",
    ]

    def __init__(self, resolution: str="Hourly"):
        self.resolution = resolution
        self.Anim = None


    @staticmethod
    def LineStyle(Style: Union[str, int]="loosely dotted"):
        """LineStyle.

        Line styles for plotting

        Parameters
        ----------
        Style : TYPE, optional
            DESCRIPTION. The default is 'loosely dotted'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if isinstance(Style, str):
            try:
                return Visualize.linestyles[Style]
            except KeyError:
                msg = (
                    " The Style name you entered-{0}-does not exist please"
                    "choose from the available styles"
                ).format(Style)
                print(msg)
                print(list(Visualize.linestyles))
        else:
            return list(Visualize.linestyles.items())[Style][1]

    @staticmethod
    def MarkerStyle(Style: int):
        """MarkerStyle.

        Marker styles for plotting

        Parameters
        ----------
        Style : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if Style > len(Visualize.MarkerStyleList) - 1:
            Style = Style % len(Visualize.MarkerStyleList)
        return Visualize.MarkerStyleList[Style]

    def GroundSurface(
        self,
        Sub,
        fromxs: Union[str, int]="",
        toxs: Union[str, int]="",
        floodplain: bool=False,
        plotlateral: bool=False,
        nxlabels: int=10,
        figsize: tuple=(20, 10),
        LateralsColor: Union[str, tuple]="red",
        LaterlasLineWidth: int=1,
        option: int=1,
        size: int=50,
    ):
        """Plot the longitudinal profile of the segment.

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        fromxs : TYPE, optional
            DESCRIPTION. The default is ''.
        toxs : TYPE, optional
            DESCRIPTION. The default is ''.
        floodplain : TYPE, optional
            DESCRIPTION. The default is False.
        plotlateral : TYPE, optional
            DESCRIPTION. The default is False.
        nxlabels: [int]
            Default is 10
        figsize: [tuple]
            Default is (20, 10)
        LateralsColor: [str, tuple]
            Defaut is "red",
        LaterlasLineWidth: [int]
            Default is 1.
        option: [int]
            Default is 1
        size: [int]
        Default is 50.

        Returns
        -------
        None.

        """
        GroundSurfacefig = plt.figure(70, figsize=figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=GroundSurfacefig)
        axGS = GroundSurfacefig.add_subplot(gs[0:2, 0:6])

        if fromxs == "":
            fromxs = Sub.xsname[0]

        if toxs == "":
            toxs = Sub.xsname[-1]

        # not the whole sub-basin
        axGS.set_xticks(list(range(fromxs, toxs)))
        axGS.set_xticklabels(list(range(fromxs, toxs)))

        axGS.set_xlim(fromxs - 1, toxs + 1)

        axGS.tick_params(labelsize=8)
        # plot dikes
        axGS.plot(
            Sub.xsname,
            Sub.crosssections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        axGS.plot(
            Sub.xsname, Sub.crosssections["zr"], "k.-", linewidth=2, label="Right Dike"
        )

        if floodplain:
            fpl = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hl"]
            )
            fpr = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hr"]
            )
            axGS.plot(Sub.xsname, fpl, "b-.", linewidth=2, label="Floodplain left")
            axGS.plot(Sub.xsname, fpr, "r-.", linewidth=2, label="Floodplain right")

        if plotlateral:
            if isinstance(Sub.LateralsTable, list) and len(Sub.LateralsTable) > 0:
                if option == 1:
                    # plot location of laterals
                    for i in range(len(Sub.LateralsTable)):
                        axGS.vlines(
                            Sub.LateralsTable[i],
                            0,
                            int(Sub.Result1D["q"].max()),
                            colors=LateralsColor,
                            linestyles="dashed",
                            linewidth=LaterlasLineWidth,
                        )
                else:
                    lat = pd.DataFrame()
                    lat["xsid"] = Sub.LateralsTable
                    lat = lat.merge(Sub.crosssections, on="xsid", how="left")

                    axGS.scatter(
                        Sub.LateralsTable,
                        lat["gl"].tolist(),
                        c=LateralsColor,
                        linewidth=LaterlasLineWidth,
                        zorder=10,
                        s=size,
                    )
            else:
                print(" Please Read the Laterals data")

        maxelevel1 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zr"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        maxelevel2 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zl"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        maxlelv = max(maxelevel1, maxelevel2)
        minlev = Sub.crosssections.loc[Sub.crosssections["xsid"] == toxs, "gl"].values
        axGS.set_ylim(minlev - 5, maxlelv + 5)

        # plot the bedlevel/baklevel
        if Sub.version == 1:
            axGS.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            axGS.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            axGS.plot(
                Sub.xsname,
                Sub.crosssections["gl"] + Sub.crosssections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if nxlabels != "":
            start, end = axGS.get_xlim()
            label_list = [int(i) for i in np.linspace(start, end, nxlabels)]
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
        fps: int=100,
        fromxs: Union[str, int]="",
        toxs: Union[str, int]="",
        fmt: str="%Y-%m-%d",
        figsize: tuple=(20, 10),
        textlocation: tuple=(1, 1),
        LateralsColor: Union[int, str]="#3D59AB",
        LaterlasLineWidth: int=1,
        xaxislabelsize: int=10,
        yaxislabelsize: int=10,
        nxlabels: int=10,
        xticklabelsize: int=8,
        Lastsegment: bool=True,
        floodplain: bool=True,
        repeat: bool=True,
    ) -> FuncAnimation:
        """WaterSurfaceProfile.

        Plot water surface profile

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object.
        start : [datetime object]
            starting date of the simulation.
        end : [datetime object]
            end date of the simulation.
        fps : [integer], optional
             It is an optional integer value that represents the delay between
             each frame in milliseconds. Its default is 100.
        fromxs : [integer], optional
            number of cross sections to be displayed before the chosen cross
            section . The default is 10.
        toxs : [integer], optional
            number of cross sections to be displayed after the chosen cross
            section . The default is 10.
        xticklabelsize: []

        nxlabels:[]

        yaxislabelsize: []

        LaterlasLineWidth: []

        xaxislabelsize:[]

        LateralsColor: []

        textlocation: []

        fmt: []

        figsize: []

        Lastsegment: [bool]
            Default is True.
        floodplain: [bool]
            Default is True.
        repeat: [bool]
            Defaut is True

        Returns
        -------




        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)

        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)
        msg = """The start date does not exist in the results, Results are
            between {0} and  {1}""".format(
            Sub.firstday, Sub.lastday
        )
        assert start in Sub.referenceindex_results, msg

        msg = """ The end date does not exist in the results, Results are
            between {0} and  {1}""".format(
            Sub.firstday, Sub.lastday
        )
        assert end in Sub.referenceindex_results, msg

        msg = """please read the boundary condition files using the
            'ReadBoundaryConditions' method """
        assert hasattr(Sub, "QBC"), msg

        msg = """ start Simulation date should be before the end simulation
            date """
        assert start < end, msg

        if Sub.from_beginning == 1:
            Period = Sub.daylist[
                np.where(Sub.referenceindex == start)[0][0] : np.where(
                    Sub.referenceindex == end
                )[0][0]
                + 1
            ]
        else:
            ii = Sub.DateToIndex(start)
            ii2 = Sub.DateToIndex(end)
            Period = list(range(ii, ii2 + 1))

        counter = [(i, j) for i in Period for j in hours]

        fig = plt.figure(60, figsize=figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 2:6])
        ax1.set_ylim(0, int(Sub.Result1D["q"].max()))

        if fromxs == "":
            # xs = 0
            # plot the whole sub-basin
            fromxs = Sub.xsname[0]
        else:
            # xs = 1
            # not the whole sub-basin
            if fromxs < Sub.xsname[0]:
                fromxs = Sub.xsname[0]

        if toxs == "":
            toxs = Sub.xsname[-1]
        else:
            if toxs > Sub.xsname[-1]:
                toxs = Sub.xsname[-1]

        ax1.set_xlim(fromxs - 1, toxs + 1)
        ax1.set_xticks(list(range(fromxs, toxs + 1)))
        ax1.set_xticklabels(list(range(fromxs, toxs + 1)))

        ax1.tick_params(labelsize=xticklabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.set_xlabel("Cross section No", fontsize=xaxislabelsize)
        ax1.set_ylabel("Discharge (m3/s)", fontsize=yaxislabelsize, labelpad=0.3)
        ax1.set_title("Sub-Basin" + " " + str(Sub.id), fontsize=15)
        ax1.legend(["Discharge"], fontsize=15)

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ax1.vlines(
                Sub.LateralsTable[i],
                0,
                int(Sub.Result1D["q"].max()),
                colors=LateralsColor,
                linestyles="dashed",
                linewidth=LaterlasLineWidth,
            )

        lat = pd.DataFrame()
        lat["xsid"] = Sub.LateralsTable
        lat = lat.merge(Sub.crosssections, on="xsid", how="left")

        lim = ax1.get_ylim()
        y = np.ones(len(Sub.LateralsTable), dtype=int) * (lim[1] - 50)
        lat = ax1.scatter(
            Sub.LateralsTable,
            y,
            c=LateralsColor,
            linewidth=LaterlasLineWidth,
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

        ax2.set_xlabel("Time", fontsize=yaxislabelsize)
        ax2.set_ylabel("Q (m3/s)", fontsize=yaxislabelsize, labelpad=0.1)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=15)

        (bc_q_line,) = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=300)
        ax2.grid()

        # h
        ax3 = fig.add_subplot(gs[0, 0])
        ax3.set_xlim(1, 25)
        ax3.set_ylim(float(Sub.HBC.min().min()), float(Sub.HBC.max().max()))

        ax3.set_xlabel("Time", fontsize=yaxislabelsize)
        ax3.set_ylabel("water level", fontsize=yaxislabelsize, labelpad=0.5)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        (bc_h_line,) = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig.add_subplot(gs[1, 0:6])

        ymax1 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zr"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        ymax2 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zl"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        ymax = max(ymax1, ymax2)
        minlev = Sub.crosssections.loc[Sub.crosssections["xsid"] == toxs, "gl"].values
        ax4.set_ylim(minlev - 5, ymax + 5)
        ax4.set_xlim(fromxs - 1, toxs + 1)
        ax4.set_xticks(list(range(fromxs, toxs + 1)))
        ax4.set_xticklabels(list(range(fromxs, toxs + 1)))

        ax4.tick_params(labelsize=xticklabelsize)
        ax4.locator_params(axis="x", nbins=nxlabels)

        ax4.plot(
            Sub.xsname,
            Sub.crosssections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        ax4.plot(
            Sub.xsname, Sub.crosssections["zr"], "k.-", linewidth=2, label="Right Dike"
        )

        if Sub.version == 1:
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"] + Sub.crosssections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if floodplain:
            fpl = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hl"]
            )
            fpr = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hr"]
            )
            ax4.plot(Sub.xsname, fpl, "b-.", linewidth=2, label="Floodplain left")
            ax4.plot(Sub.xsname, fpr, "r-.", linewidth=2, label="Floodplain right")

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(fontsize=15)
        ax4.set_xlabel("Profile", fontsize=yaxislabelsize)
        ax4.set_ylabel("Elevation m", fontsize=yaxislabelsize)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ymin = Sub.crosssections.loc[
                Sub.crosssections["xsid"] == Sub.LateralsTable[i], "gl"
            ].values[0]
            ax4.vlines(
                Sub.LateralsTable[i],
                ymin,
                ymax,
                colors=LateralsColor,
                linestyles="dashed",
                linewidth=LaterlasLineWidth,
            )

        day_text = ax4.annotate(
            "",
            xy=(
                fromxs + textlocation[0],
                Sub.crosssections.loc[Sub.crosssections["xsid"] == toxs, "gl"].values
                + textlocation[1],
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
            x = Sub.xsname
            y = Sub.Result1D.loc[Sub.Result1D["day"] == counter[i][0], "q"][
                Sub.Result1D["hour"] == counter[i][1]
            ]
            # the Saintvenant subroutine writes the
            # results of the last xs in the next segment with the current
            # segment
            if not Lastsegment:
                y = y.values[:-1]

            q_line.set_data(x, y)

            day = Sub.referenceindex.loc[counter[i][0], "date"]

            if len(Sub.LateralsTable) > 0:
                lat.set_sizes(
                    sizes=Sub.Laterals.loc[day, Sub.LateralsTable].values * 100
                )

            day_text.set_text("day = " + str(day + dt.timedelta(hours=counter[i][1])))

            y = Sub.Result1D.loc[Sub.Result1D["day"] == counter[i][0], "wl"][
                Sub.Result1D["hour"] == counter[i][1]
            ]
            # the Saintvenant subroutine writes the results
            # of the last xs in the next segment with the current segment
            if not Lastsegment:
                y = y.values[:-1]

            wl_line.set_data(x, y)

            y = (
                Sub.Result1D.loc[Sub.Result1D["day"] == counter[i][0], "h"][
                    Sub.Result1D["hour"] == counter[i][1]
                ]
                * 2
            )
            # temporary as now the Saintvenant subroutine writes the results
            # of the last xs in the next segment with the current segment
            if not Lastsegment:
                y = y.values[:-1]

            y = (
                y
                + Sub.crosssections.loc[
                    Sub.crosssections.index[len(Sub.xsname) - 1], "gl"
                ]
            )
            hLline.set_data(x, y)

            x = Sub.QBC.columns.values

            y = Sub.QBC.loc[Sub.referenceindex.loc[counter[i][0], "date"]].values
            bc_q_line.set_data(x, y)

            y = Sub.HBC.loc[Sub.referenceindex.loc[counter[i][0], "date"]].values
            bc_h_line.set_data(x, y)

            x = counter[i][1]
            y = Sub.referenceindex.loc[counter[i][0], "date"]
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
        interval: float=0.00002,
        fromxs: Union[str, int]="",
        toxs: Union[str, int]="",
        fmt: str="%Y-%m-%d",
        figsize: tuple=(20, 10),
        textlocation: tuple=(1, 1),
        LateralsColor: Union[str, tuple]="#3D59AB",
        LaterlasLineWidth: int=1,
        xaxislabelsize: int=10,
        yaxislabelsize: int=10,
        nxlabels: int=20,
        xticklabelsize: int=8,
        floodplain: bool=True,
        repeat: bool=True,
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
        fromxs : TYPE, optional
            DESCRIPTION. The default is 10.
        toxs : TYPE, optional
            DESCRIPTION. The default is 10.
        floodplain: [bool]
            Default is True.
        fmt: [str]
            Default is "%Y-%m-%d".
        figsize: [tuple]
            Default is (20, 10).
        textlocation: [tuple]
            Default is (1, 1).
        LateralsColor: [str]
            Default is "#3D59AB".
        LaterlasLineWidth: [int]
            Default is 1.
        xaxislabelsize: [int]
            Default is 10.
        yaxislabelsize: [int]
            Default is 10.
        nxlabels: [int]
            Default is 20.
        xticklabelsize: [int]
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

        if fromxs == "":
            fromxs = Sub.xsname[0]
            # toxs = Sub.xsname[-1]
        else:
            if fromxs < Sub.xsname[0]:
                fromxs = Sub.xsname[0]

            # if toxs > Sub.xsname[-1]:
                # toxs = Sub.xsname[-1]

        if toxs == "":
            toxs = Sub.xsname[-1]
        else:
            if toxs > Sub.xsname[-1]:
                toxs = Sub.xsname[-1]

        ax1.set_xlim(fromxs - 1, toxs + 1)

        ax1.set_xticks(list(range(fromxs, toxs + 1)))
        ax1.set_xticklabels(list(range(fromxs, toxs + 1)))

        ax1.tick_params(labelsize=xticklabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.set_xlabel("Cross section No", fontsize=xaxislabelsize)
        ax1.set_ylabel("Discharge (m3/s)", fontsize=yaxislabelsize, labelpad=0.5)
        ax1.set_title("Sub-Basin" + " " + str(Sub.id), fontsize=15)
        ax1.legend(["Discharge"], fontsize=15)
        ax1.set_ylim(0, int(Sub.q.max().max()))

        if Sub.version < 4:
            # ax1.set_ylim(0, int(Sub.Result1D['q'].max()))

            # plot location of laterals
            for i in range(len(Sub.LateralsTable)):
                ax1.vlines(
                    Sub.LateralsTable[i],
                    0,
                    int(int(Sub.q.max().max())),
                    colors=LateralsColor,
                    linestyles="dashed",
                    linewidth=LaterlasLineWidth,
                )

            lat = pd.DataFrame()
            lat["xsid"] = Sub.LateralsTable
            lat = lat.merge(Sub.crosssections, on="xsid", how="left")

            lim = ax1.get_ylim()
            y = np.ones(len(Sub.LateralsTable), dtype=int) * (lim[1] - 50)
            lat = ax1.scatter(
                Sub.LateralsTable,
                y,
                c=LateralsColor,
                linewidth=LaterlasLineWidth,
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
            ax2.set_ylim(0, int(Sub.QBCmin.max().max()))
        else:
            ax2.set_ylim(0, int(Sub.USBC.max()))

        ax2.set_xlabel("Time", fontsize=yaxislabelsize)
        ax2.set_ylabel("Q (m3/s)", fontsize=yaxislabelsize, labelpad=0.1)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=15)

        (bc_q_line,) = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=150)
        ax2.grid()

        # h
        ax3 = fig2.add_subplot(gs[0, 0:1])
        ax3.set_xlim(1, nstep)
        if Sub.version < 4:
            ax3.set_ylim(float(Sub.HBCmin.min().min()), float(Sub.HBCmin.max().max()))

        ax3.set_xlabel("Time", fontsize=yaxislabelsize)
        ax3.set_ylabel("water level", fontsize=yaxislabelsize, labelpad=0.5)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        (bc_h_line,) = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=150)

        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        ymax1 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zr"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        ymax2 = max(
            Sub.crosssections.loc[Sub.crosssections["xsid"] >= fromxs, "zl"][
                Sub.crosssections["xsid"] <= toxs
            ]
        )
        ymax = max(ymax1, ymax2)
        minlev = Sub.crosssections.loc[Sub.crosssections["xsid"] == toxs, "gl"].values
        ax4.set_ylim(minlev - 5, ymax + 5)
        ax4.set_xlim(fromxs - 1, toxs + 1)
        ax4.set_xticks(list(range(fromxs, toxs + 1)))

        ax4.tick_params(labelsize=xaxislabelsize)
        ax4.locator_params(axis="x", nbins=nxlabels)

        ax4.plot(
            Sub.xsname,
            Sub.crosssections["zl"],
            "k--",
            dashes=(5, 1),
            linewidth=2,
            label="Left Dike",
        )
        ax4.plot(
            Sub.xsname, Sub.crosssections["zr"], "k.-", linewidth=2, label="Right Dike"
        )

        if Sub.version == 1:
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Bankful level",
            )
        else:
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"],
                "k-",
                linewidth=5,
                label="Ground level",
            )
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["gl"] + Sub.crosssections["dbf"],
                "k",
                linewidth=2,
                label="Bankful depth",
            )

        if floodplain:
            fpl = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hl"]
            )
            fpr = (
                Sub.crosssections["gl"]
                + Sub.crosssections["dbf"]
                + Sub.crosssections["hr"]
            )
            ax4.plot(Sub.xsname, fpl, "b-.", linewidth=2, label="Floodplain left")
            ax4.plot(Sub.xsname, fpr, "r-.", linewidth=2, label="Floodplain right")

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(fontsize=10)
        ax4.set_xlabel("Profile", fontsize=10)
        ax4.set_ylabel("Elevation m", fontsize=10)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ymin = Sub.crosssections.loc[
                Sub.crosssections["xsid"] == Sub.LateralsTable[i], "gl"
            ].values[0]
            ax4.vlines(
                Sub.LateralsTable[i],
                ymin,
                ymax,
                colors=LateralsColor,
                linestyles="dashed",
                linewidth=LaterlasLineWidth,
            )

        day_text = ax4.annotate(
            "",
            xy=(
                fromxs + textlocation[0],
                Sub.crosssections.loc[Sub.crosssections["xsid"] == toxs, "gl"].values
                + textlocation[1],
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
            x = Sub.xsname
            y = Sub.q[Sub.q.index == counter[i]].values[0]
            q_line.set_data(x, y)

            # water level (ax4)
            y = Sub.h.loc[Sub.q.index == counter[i]].values[0]
            wl_line.set_data(x, y)

            day = counter[i].floor(freq="D")

            lat.set_sizes(sizes=Sub.Laterals.loc[day, Sub.LateralsTable].values * 100)

            # BC Q (ax2)

            x = Sub.QBCmin.columns.values

            y = Sub.QBCmin.loc[day].values
            bc_q_line.set_data(x, y)

            # BC H (ax3)
            y = Sub.HBCmin.loc[day].values
            bc_h_line.set_data(x, y)

            # BC Q point (ax2)
            x = ((counter[i] - day).seconds / 60) + 1
            scatter1 = ax2.scatter(x, Sub.QBCmin[x][day], s=150)
            # BC h point (ax3)
            scatter2 = ax3.scatter(x, Sub.HBCmin[x][day], s=150)

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
        xsbefore=10,
        xsafter=10,
        fmt="%Y-%m-%d",
        textlocation=2,
        xaxislabelsize=15,
        yaxislabelsize=15,
        nxlabels=50,
        plotbanhfuldepth=False,
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
        xsbefore : TYPE, optional
            DESCRIPTION. The default is 10.
        xsafter : TYPE, optional
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

        assert start in Sub.referenceindex_results, msg.format(
            Sub.referenceindex_results[0], Sub.referenceindex_results[-1]
        )

        assert end in Sub.referenceindex_results, msg.format(
            Sub.referenceindex_results[0], Sub.referenceindex_results[-1]
        )

        counter = Sub.referenceindex_results[
            np.where(Sub.referenceindex_results == start)[0][0] : np.where(
                Sub.referenceindex_results == end
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
            # ax1.set_ylabel('USBC - Q (m3/s)', fontsize=15)
            ax1.set_title("USBC - Q (m3/s)", fontsize=20)
        else:
            # ax1.set_ylabel('USBC - H (m)', fontsize=15)
            ax1.set_title("USBC - H (m)", fontsize=20)
        # ax1.legend(["Q"], fontsize=10)
        ax1.set_xlim(1, 25)
        (usbc_line,) = ax1.plot([], [], linewidth=5)
        # usbc_point = ax1.scatter([], [], s=150)
        ax1.grid()

        ax2 = fig2.add_subplot(gs[0, 1:5])
        if xs == 0:
            # plot the whole sub-basin
            ax2.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax2.set_xticks(Sub.xsname)
            ax2.set_xticklabels(Sub.xsname)

            FigureFirstXS = Sub.xsname[0]
            FigureLastXS = Sub.xsname[-1]
        else:
            # not the whole sub-basin
            FigureFirstXS = Sub.xsname[xs] - xsbefore
            if FigureFirstXS < Sub.xsname[0]:
                FigureFirstXS = Sub.xsname[0]

            FigureLastXS = Sub.xsname[xs] + xsafter
            if FigureLastXS > Sub.xsname[-1]:
                FigureLastXS = Sub.xsname[-1]

            ax2.set_xlim(FigureFirstXS, FigureLastXS)
            ax2.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax2.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        ax2.set_ylim(np.nanmin(Sub.q) - 10, int(np.nanmax(Sub.q)) + 10)
        ax2.tick_params(labelsize=xaxislabelsize)
        ax2.locator_params(axis="x", nbins=nxlabels)
        ax2.set_xlabel("Cross section No", fontsize=xaxislabelsize)
        ax2.set_title("Discharge (m3/s)", fontsize=20)
        # ax2.set_ylabel('Discharge (m3/s)', fontsize=yaxislabelsize, labelpad=0.5)
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
            # ax3.set_ylabel('DSBC', fontsize=15, labelpad=0.5)
            ax3.set_title("DSBC - Q (m3/s)", fontsize=20)
        else:
            # ax3.set_ylabel('USBC', fontsize=15, labelpad=0.5)
            ax3.set_title("DSBC - H(m)", fontsize=20)

        # ax3.legend(["WL"], fontsize=10)

        (dsbc_line,) = ax3.plot([], [], linewidth=5)
        # dsbc_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        if xs == 0:
            ax4.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax4.set_xticks(Sub.xsname)
            ymin = Sub.crosssections.loc[
                Sub.crosssections["xsid"] == FigureFirstXS, "bed level"
            ].values.min()
            ymax = Sub.crosssections.loc[
                Sub.crosssections["xsid"] == FigureFirstXS, "bed level"
            ].values.max()
            ax4.set_ylim(ymin, ymax + np.nanmax(Sub.h) + 5)
        else:
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(
                Sub.crosssections.loc[
                    Sub.crosssections["xsid"] == FigureFirstXS, "bed level"
                ].values,
                Sub.crosssections.loc[
                    Sub.crosssections["xsid"] == FigureLastXS, "zr"
                ].values
                + 5,
            )

        ax4.tick_params(labelsize=xaxislabelsize)
        ax4.locator_params(axis="x", nbins=nxlabels)

        ax4.plot(
            Sub.xsname,
            Sub.crosssections["bed level"],
            "k-",
            linewidth=5,
            label="Ground level",
        )
        if plotbanhfuldepth:
            ax4.plot(
                Sub.xsname,
                Sub.crosssections["bed level"] + Sub.crosssections["depth"],
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
            textlocation = textlocation + Sub.xsname[0]
            day_text = ax4.annotate(
                " ",
                xy=(textlocation, Sub.crosssections["bed level"].min() + 1),
                fontsize=20,
            )
        else:
            day_text = ax4.annotate(
                " ",
                xy=(
                    FigureFirstXS + textlocation,
                    Sub.crosssections.loc[
                        Sub.crosssections["xsid"] == FigureLastXS, "gl"
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
            x = Sub.xsname

            y = Sub.q[np.where(Sub.referenceindex_results == counter[i])[0][0], :]
            q_line.set_data(x, y)

            # water level (ax4)
            y = Sub.wl[np.where(Sub.referenceindex_results == counter[i])[0][0], :]
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
            # y = Sub.referenceindex.loc[counter[i][0], 'date']
            # ax2.scatter(x, Sub.QBC[x][y])

            # # BC Q point (ax2)
            # x = ((counter[i] - dt.datetime(counter[i].year, counter[i].month, counter[i].day)).seconds / 60) + 1
            # y = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            # ax2.scatter(x, Sub.USBC[x][y])

            return (
                q_line,
                wl_line,
                day_text,
                usbc_line,
                dsbc_line,
            )  # ax2.scatter(x, Sub.USBC[x][y], s=150),

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

    def CrossSections(
        self,
        Sub,
        fromxs: Union[str, int]="",
        toxs: Union[str, int]="",
        xsrows: int=3,
        xscolumns: int=3,
        bedlevel: bool=False,
        titlesize: int=15,
        textsize: int=15,
        figsize: tuple=(18, 10),
        linewidth: int=6,
        samescale: bool=False,
        textspacing: List[tuple]=[(1, 1), (1, 2)],
        plottingoption: int=1,
        plotannotation: bool=True,
    ):
        """CrossSections.

        Plot cross sections of a river segment.

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object..
        fromxs : TYPE, optional
            DESCRIPTION. The default is ''.
        toxs : TYPE, optional
            DESCRIPTION. The default is ''.
        xsrows : TYPE, optional
            DESCRIPTION. The default is 3.
        xscolumns : TYPE, optional
            DESCRIPTION. The default is 3.
        bedlevel : TYPE, optional
            DESCRIPTION. The default is False.
        titlesize : TYPE, optional
            DESCRIPTION. The default is 15.
        textsize : TYPE, optional
            DESCRIPTION. The default is 15.
        figsize : TYPE, optional
            DESCRIPTION. The default is (18, 10).
        linewidth : TYPE, optional
            DESCRIPTION. The default is 6.
        plottingoption : [integer]
            1 if you want to plot the whole cross-section, 2 to execlude the
            dikes(river bankfull area and floodplain will be plotted),
            3 to plot only the bankfull area.
        samescale: [bool]
            Default is False.
        textspacing: [tuple]
            Default is [(1, 1), (1, 2)].
        plotannotation: [bool]
            Default is True.

        Returns
        -------
        None.

        """
        if fromxs == "":
            startxs_ind = 0
        else:
            startxs_ind = Sub.xsname.index(fromxs)

        if toxs == "":
            endxs_ind = Sub.xsno - 1
        else:
            endxs_ind = Sub.xsname.index(toxs)

        names = list(range(1, 17))
        XSS = pd.DataFrame(
            columns=names, index=Sub.crosssections.loc[startxs_ind:endxs_ind, "xsid"]
        )

        # calculate the vertices of the cross sections
        for i in range(startxs_ind, endxs_ind + 1):
            ind = XSS.index[i - startxs_ind]
            ind2 = Sub.crosssections.index[i]

            XSS[1].loc[XSS.index == ind] = 0
            XSS[2].loc[XSS.index == ind] = 0

            bl = Sub.crosssections["bl"].loc[Sub.crosssections.index == ind2].values[0]
            b = Sub.crosssections["b"].loc[Sub.crosssections.index == ind2].values[0]
            br = Sub.crosssections["br"].loc[Sub.crosssections.index == ind2].values[0]

            XSS[3].loc[XSS.index == ind] = bl
            XSS[4].loc[XSS.index == ind] = bl
            XSS[5].loc[XSS.index == ind] = bl + b
            XSS[6].loc[XSS.index == ind] = bl + b
            XSS[7].loc[XSS.index == ind] = bl + b + br
            XSS[8].loc[XSS.index == ind] = bl + b + br

            gl = Sub.crosssections["gl"].loc[Sub.crosssections.index == ind2].values[0]

            if bedlevel:
                subtract = 0
            else:
                subtract = gl

            zl = Sub.crosssections["zl"].loc[Sub.crosssections.index == ind2].values[0]
            zr = Sub.crosssections["zr"].loc[Sub.crosssections.index == ind2].values[0]

            if Sub.version > 1:
                dbf = (
                    Sub.crosssections["dbf"]
                    .loc[Sub.crosssections.index == ind2]
                    .values[0]
                )

            hl = Sub.crosssections["hl"].loc[Sub.crosssections.index == ind2].values[0]
            hr = Sub.crosssections["hr"].loc[Sub.crosssections.index == ind2].values[0]

            XSS[9].loc[XSS.index == ind] = zl - subtract

            if Sub.version == 1:
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
        figno = int(math.ceil(xsplot / (xscolumns * xsrows)))

        ind2 = startxs_ind
        ind = XSS.index[ind2 - startxs_ind]
        for i in range(figno):

            # fig = plt.figure(1000 + i, figsize=figsize)
            # -----------------
            if samescale:
                sharex = True
                sharey = True
            else:
                sharex = False
                sharey = False

            fig, ax_XS = plt.subplots(
                ncols=xscolumns,
                nrows=xsrows,
                figsize=figsize,
                sharex=sharex,
                sharey=sharey,
            )
            # gs = gridspec.GridSpec(xsrows, xscolumns)

            for j in range(xsrows):
                for k in range(xscolumns):
                    if ind2 <= endxs_ind:
                        XsId = Sub.crosssections["xsid"][Sub.crosssections.index[ind2]]
                        xcoord = (
                            XSS[names[0:8]].loc[XSS.index == ind].values.tolist()[0]
                        )
                        ycoord = (
                            XSS[names[8:16]].loc[XSS.index == ind].values.tolist()[0]
                        )
                        b = (
                            Sub.crosssections["b"]
                            .loc[Sub.crosssections["xsid"] == ind]
                            .values[0]
                        )
                        bl = (
                            Sub.crosssections["bl"]
                            .loc[Sub.crosssections["xsid"] == ind]
                            .values[0]
                        )
                        gl = (
                            Sub.crosssections["gl"]
                            .loc[Sub.crosssections["xsid"] == ind]
                            .values[0]
                        )

                        # ax_XS = fig.add_subplot(gs[j, k])
                        if plottingoption == 1:
                            ax_XS[j, k].plot(xcoord, ycoord, linewidth=linewidth)
                            x = textspacing[0][0]
                            x1 = textspacing[1][0]
                        elif plottingoption == 2:
                            ax_XS[j, k].plot(
                                xcoord[1:-1], ycoord[1:-1], linewidth=linewidth
                            )
                            x = textspacing[0][0] + bl
                            x1 = textspacing[1][0] + bl
                        else:
                            ax_XS[j, k].plot(
                                xcoord[2:-2], ycoord[2:-2], linewidth=linewidth
                            )
                            x = textspacing[0][0] + bl
                            x1 = textspacing[1][0] + bl

                        ax_XS[j, k].title.set_text("xs ID = " + str(XsId))
                        ax_XS[j, k].title.set_fontsize(titlesize)

                        if samescale:
                            # when sharex and sharey are true the labels
                            # disappear so set thier visability to true
                            ax_XS[j, k].xaxis.set_tick_params(labelbottom=True)
                            ax_XS[j, k].yaxis.set_tick_params(labelbottom=True)

                        if plotannotation:
                            if Sub.version > 1:
                                dbf = (
                                    Sub.crosssections["dbf"]
                                    .loc[Sub.crosssections["xsid"] == ind]
                                    .values[0]
                                )
                                b = (
                                    Sub.crosssections["b"]
                                    .loc[Sub.crosssections["xsid"] == ind]
                                    .values[0]
                                )

                                if bedlevel:
                                    ax_XS[j, k].annotate(
                                        "dbf=" + str(round(dbf, 2)),
                                        xy=(x, gl + textspacing[0][1]),
                                        fontsize=textsize,
                                    )

                                    ax_XS[j, k].annotate(
                                        "b=" + str(round(b, 2)),
                                        xy=(x1, gl + textspacing[1][1]),
                                        fontsize=textsize,
                                    )
                                else:

                                    ax_XS[j, k].annotate(
                                        "dbf=" + str(round(dbf, 2)),
                                        xy=(x, textspacing[0][1]),
                                        fontsize=textsize,
                                    )

                                    ax_XS[j, k].annotate(
                                        "b=" + str(round(b, 2)),
                                        xy=(x1, textspacing[1][1]),
                                        fontsize=textsize,
                                    )

                        ind2 = ind2 + 1
                        ind = ind + 1

            plt.subplots_adjust(
                wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96
            )

        return fig, ax_XS


    def Plot1minProfile(
        self, Sub, date: str, xaxislabelsize: int=10, nxlabels: int=50, fmt: str="%Y-%m-%d"
    ):
        """Plot1minProfile.

        Plot water surface profile for 1 min data.

        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        date : TYPE
            DESCRIPTION.
        xaxislabelsize : TYPE, optional
            DESCRIPTION. The default is 10.
        nxlabels : TYPE, optional
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
        ax1.set_xticks(Sub.xsname)
        ax1.tick_params(labelsize=xaxislabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.grid()


    @staticmethod
    def SaveProfileAnimation(Anim, Path="", fps=3, ffmpegPath=""):
        """SaveProfileAnimation.

        save video animation
        available extentions .mov, .gif, .avi, .mp4

        Parameters
        ----------
        Anim : TYPE
            DESCRIPTION.
        Path : TYPE, optional
            DESCRIPTION. The default is ''.
        fps : TYPE, optional
            DESCRIPTION. The default is 3.
        ffmpegPath : TYPE, optional
            DESCRIPTION. The default is ''.

        in order to save a video using matplotlib you have to download ffmpeg
        from https://ffmpeg.org/ and define this path to matplotlib

        import matplotlib as mpl
        mpl.rcParams['animation.ffmpeg_path'] = "path where you saved the
                                                ffmpeg.exe/ffmpeg.exe"

        Returns
        -------
        None.

        """
        message = """
            please visit https://ffmpeg.org/ and download a version of ffmpeg
            compitable with your operating system, and copy the content of the
            folder and paste it in the "c:/user/.matplotlib/ffmpeg-static/"
            """
        if ffmpegPath == "":
            ffmpegPath = os.getenv("HOME") + "/.matplotlib/ffmpeg-static/bin/ffmpeg.exe"
            assert os.path.exists(ffmpegPath), "{0}".format(message)

        mpl.rcParams["animation.ffmpeg_path"] = ffmpegPath

        Ext = Path.split(".")[1]
        if Ext == "gif":
            msg = """ please enter a valid path to save the animation"""
            assert len(Path) >= 1 and Path.endswith(".gif"), "{0}".format(msg)

            writergif = animation.PillowWriter(fps=fps)
            Anim.save(Path, writer=writergif)
        else:
            try:
                if Ext == "avi" or Ext == "mov":
                    writervideo = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    Anim.save(Path, writer=writervideo)
                elif Ext == "mp4":
                    writermp4 = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    Anim.save(Path, writer=writermp4)

            except FileNotFoundError:
                msg = """ please visit https://ffmpeg.org/ and download a
                    version of ffmpeg compitable with your operating system, for
                    more details please check the method definition
                    """
                print("{0}".format(msg))


    def SaveAnimation(anim, VideoFormat="gif", Path="", SaveFrames=20):
        """SaveAnimation.

        Save animation

        Parameters
        ----------
        anim : TYPE
            DESCRIPTION.
        VideoFormat : TYPE, optional
            DESCRIPTION. The default is "gif".
        Path : TYPE, optional
            DESCRIPTION. The default is ''.
        SaveFrames : TYPE, optional
            DESCRIPTION. The default is 20.

        in order to save a video using matplotlib you have to download ffmpeg from
        https://ffmpeg.org/ and define this path to matplotlib

        import matplotlib as mpl
        mpl.rcParams['animation.ffmpeg_path'] = "path where you saved the ffmpeg.exe/ffmpeg.exe"

        Returns
        -------
        None.

        """
        ffmpegPath = os.getenv("HOME") + "/.matplotlib/ffmpeg-static/bin/ffmpeg.exe"

        message = """
        please visit https://ffmpeg.org/ and download a version of ffmpeg compitable
        with your operating system, and copy the content of the folder and paste it
        in the "c:/user/.matplotlib/ffmpeg-static/"
        """
        assert os.path.exists(ffmpegPath), message

        mpl.rcParams["animation.ffmpeg_path"] = ffmpegPath

        if VideoFormat == "gif":
            writergif = animation.PillowWriter(fps=SaveFrames)
            anim.save(Path, writer=writergif)
        else:
            try:
                if VideoFormat == "avi" or VideoFormat == "mov":
                    writervideo = animation.FFMpegWriter(fps=SaveFrames, bitrate=1800)
                    anim.save(Path, writer=writervideo)
                elif VideoFormat == "mp4":
                    writermp4 = animation.FFMpegWriter(fps=SaveFrames, bitrate=1800)
                    anim.save(Path, writer=writermp4)
            except FileNotFoundError:
                print(
                    "please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system, for more details please check the method definition"
                )


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
        Vis.Histogram(Val1, val2,2,figsize=(5.5,4.5), color1='#27408B',
                        xlabel = 'Inundation Depth (m)', ylabel = 'Frequency', legend_size = 15,
                         fontsize=15, labelsize = 15, Axisfontsize = 11,
                         legend = ['RIM1.0', 'RIM2.0'], pdf = False, Save = False,
                         name = str(Event1.EventIndex.loc[EndInd,'id']))

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
            #         plt.legend(kwargs['legend'],fontsize = int(kwargs['legend_size']))
            #     if key == 'xlabel':
            #         plt.xlabel(kwargs['xlabel'])
            #     if key == 'ylabel':
            #         plt.ylabel(kwargs['ylabel'])
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']

        elif NoAxis == 2:
            fig, ax1 = plt.subplots(figsize=Visualize.FigureDefaultOptions["figsize"])
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
                    linewidth=Visualize.FigureDefaultOptions["linewidth"],
                    label="RIM1.0 pdf",
                )
                ax2.plot(
                    d2,
                    pdf_fitted2,
                    "-.",
                    color=Visualize.FigureDefaultOptions["color2"],
                    linewidth=Visualize.FigureDefaultOptions["linewidth"],
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
            # ax2.set_ylabel("Frequency", fontsize = 15)
            # ax2.yaxis.label.set_color(color2)

            # ax2.tick_params(axis='y', color = color2)
            # plt.title("Sub-Basin = " + str(Subid), fontsize = 15)

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
            # if key == 'labelsize':
            ax1.set_xlabel(
                Visualize.FigureDefaultOptions["xlabel"],
                fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
            )
            ax1.set_ylabel(
                Visualize.FigureDefaultOptions["ylabel"],
                fontsize=Visualize.FigureDefaultOptions["AxisLabelSize"],
            )
            # if key == 'fontsize':
            plt.rcParams.update(
                {"font.size": int(Visualize.FigureDefaultOptions["Axisfontsize"])}
            )

            # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fontsize = 15)

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
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")


class MidpointNormalize(colors.Normalize):
    """MidpointNormalize.

    !TODO needs docs

    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """MidpointNormalize.

        ! TODO needs docs

        Parameters
        ----------
        value : TYPE
            DESCRIPTION.
        clip : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))