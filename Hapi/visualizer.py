"""
Created on Sat Mar 14 16:36:01 2020

@author: mofarrag
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import animation
import numpy as np
import pandas as pd
import datetime as dt
import math
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from collections import OrderedDict

from Hapi.statistics.statisticaltools import StatisticalTools as ST

hours = list(range(1, 25))


class Visualize:
    """
    ==================
        Visualize
    ==================
    Visualize class contains different method to animate water surface profile
    spatial data change over time, styling methods for plotting

    Methods:
        1- AnimateArray
        2- CrossSections
        3- WaterSurfaceProfile
        4- CrossSections
    """

    FigureDefaultOptions = dict(ylabel='', xlabel='',
                                legend='', legend_size=10, figsize=(10, 8),
                                labelsize=10, fontsize=10, name='hist.tif',
                                color1='#3D59AB', color2="#DC143C", linewidth=3,
                                Axisfontsize=15
                                )

    linestyles = OrderedDict([('solid', (0, ())),  # 0
                              ('loosely dotted', (0, (1, 10))),  # 1
                              ('dotted', (0, (1, 5))),  # 2
                              ('densely dotted', (0, (1, 1))),  # 3
                              ('loosely dashed', (0, (5, 10))),  # 4
                              ('dashed', (0, (5, 5))),  # 5
                              ('densely dashed', (0, (5, 1))),  # 6
                              ('loosely dashdotted', (0, (3, 10, 1, 10))),  # 7
                              ('dashdotted', (0, (3, 5, 1, 5))),  # 8
                              ('densely dashdotted', (0, (3, 1, 1, 1))),  # 9
                              ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),  # 10
                              ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),  # 11
                              ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),  # 12
                              ('densely dashdotdottededited', (0, (6, 1, 1, 1, 1, 1)))])  # 13

    MarkerStyleList = ['--o', ':D', '-.H', '--x', ':v', '--|', '-+', '-^', '--s', '-.*', '-.h']

    def __init__(self, resolution="Hourly"):
        self.resolution = "Hourly"

    @staticmethod
    def LineStyle(Style='loosely dotted'):

        if type(Style) == str:
            try:
                return Visualize.linestyles[Style]
            except KeyError:
                print("The Style name you entered-" + Style + "-does not exist please choose from the available styles")
                print(list(Visualize.linestyles))
        else:
            return list(Visualize.linestyles.items())[Style][1]

    @staticmethod
    def MarkerStyle(Style):
        if Style > len(Visualize.MarkerStyleList) - 1:
            Style = Style % len(Visualize.MarkerStyleList)
        return Visualize.MarkerStyleList[Style]

    def GroundSurface(self, Sub, XSID='', xsbefore=10, xsafter=10,
                      FloodPlain=False, PlotLateral=False):

        if XSID == '':
            xs = 0
        else:
            xs = np.where(Sub.crosssections['xsid'] == XSID)[0][0]

        GroundSurfacefig = plt.figure(70, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=GroundSurfacefig)
        axGS = GroundSurfacefig.add_subplot(gs[0:2, 0:6])

        if xs == 0:
            axGS.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            axGS.set_xticks(Sub.xsname)
        else:
            # not the whole sub-basin
            FigureFirstXS = Sub.xsname[xs] - xsbefore
            FigureLastXS = Sub.xsname[xs] + xsafter
            axGS.set_xlim(FigureFirstXS, FigureLastXS)

            axGS.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            axGS.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        axGS.tick_params(labelsize=8)
        # plot dikes
        axGS.plot(Sub.xsname, Sub.crosssections['zl'], 'k--', dashes=(5, 1), linewidth=2, label='Left Dike')
        axGS.plot(Sub.xsname, Sub.crosssections['zr'], 'k.-', linewidth=2, label='Right Dike')

        if FloodPlain:
            fpl = Sub.crosssections['gl'] + Sub.crosssections['dbf'] + Sub.crosssections['hl']
            fpr = Sub.crosssections['gl'] + Sub.crosssections['dbf'] + Sub.crosssections['hr']
            axGS.plot(Sub.xsname, fpl, 'b-.', linewidth=2, label='Floodplain left')
            axGS.plot(Sub.xsname, fpr, 'r-.', linewidth=2, label='Floodplain right')

        if PlotLateral:
            if hasattr(self, 'LateralsTable'):
                print('x')
                # axGS.plot(Sub.xsname, fpr,'r-.',linewidth = 2, label = 'Floodplain right')
            else:
                print(" Please Read the Laterals data")
        # plot the bedlevel/baklevel
        if Sub.Version == 1:
            axGS.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Bankful level')
        else:
            axGS.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Ground level')
            axGS.plot(Sub.xsname, Sub.crosssections['gl'] + Sub.crosssections['dbf'], 'k', linewidth=2,
                      label='Bankful depth')

        axGS.set_title("Water surface Profile Simulation SubID = " + str(Sub.ID), fontsize=15)
        axGS.legend(fontsize=15)
        axGS.set_xlabel("Profile", fontsize=15)
        axGS.set_ylabel("Elevation m", fontsize=15)
        axGS.grid()

        GroundSurfacefig.tight_layout()

    def WaterSurfaceProfile(self, Sub, start, end, interval=100, xs=0,
                            xsbefore=10, xsafter=10, fmt="%Y-%m-%d", textlocation=2,
                            LateralsColor='red', LaterlasLineWidth=1, xaxislabelsize=15,
                            yaxislabelsize=15, nxlabels=50):
        """

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object.
        plotstart : [datetime object]
            starting date of the simulation.
        plotend : [datetime object]
            end date of the simulation.
        interval : [integer], optional
             It is an optional integer value that represents the delay between
             each frame in milliseconds. Its default is 100.
        xs : [integer], optional
            order of a specific cross section to plot the data animation around it. The default is 0.
        xsbefore : [integer], optional
            number of cross sections to be displayed before the chosen cross section . The default is 10.
        xsafter : [integer], optional
            number of cross sections to be displayed after the chosen cross section . The default is 10.
        Save : [Boolen/string]
            different formats to save the animation 'gif', 'avi', 'mov', 'mp4'.The default is False
        Path : [String]
            Path where you want to save the animation, you have to include the
            extension at the end of the path.
        fps : [integer]
            numper of frames per second

        in order to save a video using matplotlib you have to download ffmpeg from
            https://ffmpeg.org/ and define this path to matplotlib

        import matplotlib as mpl
        mpl.rcParams['animation.ffmpeg_path'] = "path where you saved the ffmpeg.exe/ffmpeg.exe"

        Returns
        -------
        TYPE

        """

        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt)

        assert start in Sub.referenceindex_results, 'The start date does not exist in the results, Results are between ' + str(
            Sub.firstday) + " and " + str(Sub.LastDay)
        assert end in Sub.referenceindex_results, 'The end date does not exist in the results, Results are between ' + str(
            Sub.firstday) + " and " + str(Sub.LastDay)

        assert hasattr(Sub, 'QBC'), "please read the boundary condition files using the 'ReadBoundaryConditions' method"

        assert start < end, "start Simulation date should be before the end simulation date "

        if Sub.from_beginning == 1:
            Period = Sub.Daylist[
                     np.where(Sub.referenceindex == start)[0][0]:np.where(Sub.referenceindex == end)[0][0] + 1]
        else:
            ii = Sub.referenceindex.index[np.where(Sub.referenceindex == start)[0][0]]
            ii2 = Sub.referenceindex.index[np.where(Sub.referenceindex == end)[0][0]]
            Period = list(range(ii, ii2 + 1))

        counter = [(i, j) for i in Period for j in hours]

        fig = plt.figure(60, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)
        ax1 = fig.add_subplot(gs[0, 2:6])
        ax1.set_ylim(0, int(Sub.Result1D['q'].max()))

        if xs == 0:
            # plot the whole sub-basin
            ax1.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax1.set_xticks(Sub.xsname)
            ax1.set_xticklabels(Sub.xsname)

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

            ax1.set_xlim(FigureFirstXS, FigureLastXS)

            ax1.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax1.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        ax1.tick_params(labelsize=xaxislabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.tick_params(labelsize=6)
        ax1.set_xlabel('Cross section No', fontsize=xaxislabelsize)
        ax1.set_ylabel('Discharge (m3/s)', fontsize=yaxislabelsize, labelpad=0.5)
        ax1.set_title('Sub-Basin' + ' ' + str(Sub.ID), fontsize=15)
        ax1.legend(["Discharge"], fontsize=15)

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ax1.vlines(Sub.LateralsTable[i], 0, int(Sub.Result1D['q'].max()),
                       colors=LateralsColor, linestyles='dashed', linewidth=LaterlasLineWidth)

        q_line, = ax1.plot([], [], linewidth=5)
        ax1.grid()

        ### BC
        # Q
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(1, 25)
        ax2.set_ylim(0, int(Sub.QBC.max().max()) + 1)

        ax2.set_xlabel('Time', fontsize=15)
        ax2.set_ylabel('Q (m3/s)', fontsize=15, labelpad=0.1)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=15)

        bc_q_line, = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=300)
        ax2.grid()

        # h
        ax3 = fig.add_subplot(gs[0, 0])
        ax3.set_xlim(1, 25)
        ax3.set_ylim(float(Sub.HBC.min().min()), float(Sub.HBC.max().max()))

        ax3.set_xlabel('Time', fontsize=15)
        ax3.set_ylabel('water level', fontsize=15, labelpad=0.5)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        bc_h_line, = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig.add_subplot(gs[1, 0:6])

        if xs == 0:
            ax4.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax4.set_xticks(Sub.xsname)
        else:
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'gl'].values,
                         Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureLastXS, 'zr'].values + 5)

        ax4.tick_params(labelsize=8)
        ax4.plot(Sub.xsname, Sub.crosssections['zl'], 'k--', dashes=(5, 1), linewidth=2, label='Left Dike')
        ax4.plot(Sub.xsname, Sub.crosssections['zr'], 'k.-', linewidth=2, label='Right Dike')

        if Sub.Version == 1:
            ax4.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Bankful level')
        else:
            ax4.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Ground level')
            ax4.plot(Sub.xsname, Sub.crosssections['gl'] + Sub.crosssections['dbf'], 'k', linewidth=2,
                     label='Bankful depth')

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(fontsize=15)
        ax4.set_xlabel("Profile", fontsize=15)
        ax4.set_ylabel("Elevation m", fontsize=15)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ymin = Sub.crosssections.loc[Sub.crosssections['xsid'] == Sub.LateralsTable[i], 'gl'].values[0]
            ymax1 = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'zl'].values[0]
            ymax2 = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'zr'].values[0]
            ax4.vlines(Sub.LateralsTable[i], ymin, max(ymax1, ymax2), colors=LateralsColor,
                       linestyles='dashed', linewidth=LaterlasLineWidth)

        if xs == 0:
            day_text = ax4.annotate('Begining', xy=(Sub.xsname[0], Sub.crosssections['gl'].min()), fontsize=20)
        else:
            day_text = ax4.annotate('Begining', xy=(FigureFirstXS + textlocation,
                                                    Sub.crosssections.loc[
                                                        Sub.crosssections['xsid'] == FigureLastXS, 'gl'].values + 1),
                                    fontsize=20)

        wl_line, = ax4.plot([], [], linewidth=5)
        hLline, = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()

        def init_q():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            hLline.set_data([], [])
            day_text.set_text('')

            bc_q_line.set_data([], [])
            bc_h_line.set_data([], [])
            bc_q_point
            bc_h_point

            return q_line, wl_line, hLline, day_text, bc_q_line, bc_h_line, bc_q_point, bc_h_point

        # animation function. this is called sequentially
        def animate_q(i):
            x = Sub.xsname
            y = Sub.Result1D.loc[Sub.Result1D['day'] == counter[i][0], 'q'][Sub.Result1D['hour'] == counter[i][1]]
            # temporary as now the Saintvenant subroutine writes the results of the last xs in the next
            # segment with the current segment
            y = y.values[:-1]

            q_line.set_data(x, y)

            day = Sub.referenceindex.loc[counter[i][0], 'date']
            day_text.set_text('day = ' + str(day + dt.timedelta(hours=counter[i][1])))

            y = Sub.Result1D.loc[Sub.Result1D['day'] == counter[i][0], 'wl'][Sub.Result1D['hour'] == counter[i][1]]
            # temporary as now the Saintvenant subroutine writes the results of the last xs in the next
            # segment with the current segment
            y = y.values[:-1]
            wl_line.set_data(x, y)

            y = Sub.Result1D.loc[Sub.Result1D['day'] == counter[i][0], 'h'][Sub.Result1D['hour'] == counter[i][1]] * 2
            # temporary as now the Saintvenant subroutine writes the results of the last xs in the next
            # segment with the current segment
            y = y.values[:-1]
            y = y + Sub.crosssections.loc[Sub.crosssections.index[len(Sub.xsname) - 1], 'gl']
            hLline.set_data(x, y)

            x = Sub.QBC.columns.values

            y = Sub.QBC.loc[Sub.referenceindex.loc[counter[i][0], 'date']].values
            bc_q_line.set_data(x, y)

            y = Sub.HBC.loc[Sub.referenceindex.loc[counter[i][0], 'date']].values
            bc_h_line.set_data(x, y)

            x = counter[i][1]
            y = Sub.referenceindex.loc[counter[i][0], 'date']
            ax2.scatter(x, Sub.QBC[x][y])
            ax3.scatter(x, Sub.QBC[x][y])

            return q_line, wl_line, hLline, day_text, bc_q_line, bc_h_line, ax2.scatter(x, Sub.QBC[x][y],
                                                                                      s=300), ax3.scatter(x,
                                                                                                          Sub.HBC[x][y],
                                                                                                          s=300)

        # plt.tight_layout()

        Anim = animation.FuncAnimation(fig, animate_q, init_func=init_q, frames=np.shape(counter)[0],
                                       interval=interval, blit=True)
        self.Anim = Anim
        return Anim

    def WaterSurfaceProfile1Min(self, Sub, plotstart, plotend, interval=0.00002, xs=0,
                                xsbefore=10, xsafter=10, fmt="%Y-%m-%d", textlocation=2,
                                LateralsColor='red', LaterlasLineWidth=1, xaxislabelsize=15,
                                yaxislabelsize=15, nxlabels=50):
        """




        Parameters
        ----------
        Sub : TYPE
            DESCRIPTION.
        plotstart : TYPE
            DESCRIPTION.
        plotend : TYPE
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

        plotstart = dt.datetime.strptime(plotstart, fmt)
        plotend = dt.datetime.strptime(plotend, fmt) - dt.timedelta(minutes=int(Sub.dt / 60))

        assert plotstart in Sub.Hmin.index, (
                "plot start date in not in the 1min results, the results starts from " + str(
            Sub.Hmin.index[0]) + " - and ends on " + str(Sub.Hmin.index[-1]))
        assert plotend in Sub.Hmin.index, ("plot end date in not in the 1min results, the results starts from " + str(
            Sub.Hmin.index[0]) + " - and ends on " + str(Sub.Hmin.index[-1]))

        counter = Sub.Hmin.index[np.where(Sub.Hmin.index == plotstart)[0][0]:np.where(Sub.Hmin.index == plotend)[0][0]]

        fig2 = plt.figure(20, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig2)

        ax1 = fig2.add_subplot(gs[0, 2:6])

        if xs == 0:
            # plot the whole sub-basin
            ax1.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax1.set_xticks(Sub.xsname)
            ax1.set_xticklabels(Sub.xsname)

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

            ax1.set_xlim(FigureFirstXS, FigureLastXS)

            ax1.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax1.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))
        if Sub.Version < 4:
            ax1.set_ylim(0, int(Sub.Result1D['q'].max()))
        else:
            ax1.set_ylim(0, int(Sub.Qmin.max().max()))

        ax1.tick_params(labelsize=xaxislabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.set_xlabel('Cross section No', fontsize=xaxislabelsize)
        ax1.set_ylabel('Discharge (m3/s)', fontsize=yaxislabelsize, labelpad=0.5)

        ax1.legend(["Discharge"], fontsize=15)

        if Sub.Version < 4:
            # plot location of laterals
            for i in range(len(Sub.LateralsTable)):
                ax1.vlines(Sub.LateralsTable[i], 0, int(Sub.Result1D['q'].max()),
                           colors=LateralsColor, linestyles='dashed', linewidth=LaterlasLineWidth)

        q_line, = ax1.plot([], [],
                           linewidth=5)  # Sub.Result1D['q'][Sub.Result1D['day'] == Sub.Result1D['day'][1]][Sub.Result1D['hour'] == 1]
        ax1.grid()

        ### BC
        # Q
        ax2 = fig2.add_subplot(gs[0, 1:2])
        ax2.set_xlim(1, 1440)
        if Sub.Version < 4:
            ax2.set_ylim(0, int(Sub.QBCmin.max()))
        else:
            ax2.set_ylim(0, int(Sub.USBC.max()))
        # ax2.set_xticks(xsname)
        # ax2.set_xticklabels(xsname)
        # if len(xsname) > 35:
        #    ax2.tick_params(labelsize= 5)
        # else :
        #    ax2.tick_params(labelsize= 8)

        ax2.set_xlabel('Time', fontsize=15)
        ax2.set_ylabel('Discharge (m3/s)', fontsize=15)
        ax2.set_title("BC - Q", fontsize=20)
        ax2.legend(["Q"], fontsize=10)

        bc_q_line, = ax2.plot([], [], linewidth=5)
        bc_q_point = ax2.scatter([], [], s=150)
        ax2.grid()

        # h
        ax3 = fig2.add_subplot(gs[0, 0:1])
        ax3.set_xlim(1, 1440)
        if Sub.Version < 4:
            ax3.set_ylim(float(Sub.HBC.min().min()), float(Sub.HBC.max().max()))

        # ax3.set_xticks(xsname)
        # ax3.set_xticklabels(xsname)
        # if len(xsname) > 35:
        #    ax3.tick_params(labelsize= 5)
        # else :
        #    ax3.tick_params(labelsize= 8)

        ax3.set_xlabel('Time', fontsize=15)
        ax3.set_ylabel('water level', fontsize=15)
        ax3.set_title("BC - H", fontsize=20)
        ax3.legend(["WL"], fontsize=10)

        bc_h_line, = ax3.plot([], [], linewidth=5)
        bc_h_point = ax3.scatter([], [], s=150)

        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        if xs == 0:
            ax4.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax4.set_xticks(Sub.xsname)
        else:
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'gl'].values,
                         Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureLastXS, 'zr'].values + 5)

        ax4.tick_params(labelsize=xaxislabelsize)
        ax4.locator_params(axis="x", nbins=nxlabels)

        ax4.plot(Sub.xsname, Sub.crosssections['zl'], 'k--', dashes=(5, 1), linewidth=2, label='Left Dike')
        ax4.plot(Sub.xsname, Sub.crosssections['zr'], 'k.-', linewidth=2, label='Right Dike')

        if Sub.Version == 1:
            ax4.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Bankful level')
        else:
            ax4.plot(Sub.xsname, Sub.crosssections['gl'], 'k-', linewidth=5, label='Ground level')
            ax4.plot(Sub.xsname, Sub.crosssections['gl'] + Sub.crosssections['dbf'], 'k', linewidth=2,
                     label='Bankful depth')

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(['Ground level', 'Left Dike', 'Right Dike', 'Bankful depth'], fontsize=10)
        ax4.set_xlabel("Profile", fontsize=10)
        ax4.set_ylabel("Elevation m", fontsize=10)
        ax4.grid()

        # plot location of laterals
        for i in range(len(Sub.LateralsTable)):
            ymin = Sub.crosssections.loc[Sub.crosssections['xsid'] == Sub.LateralsTable[i], 'gl'].values[0]
            ymax1 = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'zl'].values[0]
            ymax2 = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'zr'].values[0]
            ax4.vlines(Sub.LateralsTable[i], ymin, max(ymax1, ymax2), colors=LateralsColor,
                       linestyles='dashed', linewidth=LaterlasLineWidth)

        if xs == 0:
            day_text = ax4.annotate('Begining', xy=(Sub.xsname[0], Sub.crosssections['gl'].min()), fontsize=20)
        else:
            day_text = ax4.annotate('Begining', xy=(FigureFirstXS + textlocation,
                                                    Sub.crosssections.loc[
                                                        Sub.crosssections['xsid'] == FigureLastXS, 'gl'].values + 1),
                                    fontsize=20)

        wl_line, = ax4.plot([], [], linewidth=5)
        hLline, = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()

        # animation
        def init_min():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            day_text.set_text('')
            bc_q_line.set_data([], [])
            bc_h_line.set_data([], [])
            bc_q_point
            bc_h_point
            return q_line, wl_line, bc_q_line, bc_h_line, bc_q_point, bc_h_point, day_text

        # animation function. this is called sequentially
        def animate_min(i):
            day_text.set_text('Date = ' + str(counter[i]))
            # discharge (ax1)
            x = Sub.xsname
            y = Sub.Qmin[Sub.Qmin.index == counter[i]].values[0]
            q_line.set_data(x, y)

            # water level (ax4)
            y = Sub.Hmin.loc[Sub.Qmin.index == counter[i]].values[0]
            wl_line.set_data(x, y)
            # BC Q (ax2)

            x = Sub.QBCmin.columns.values
            y = Sub.QBCmin.loc[dt.datetime(counter[i].year, counter[i].month, counter[i].day)].values
            bc_q_line.set_data(x, y)

            # BC H (ax3)
            y = Sub.HBCmin.loc[dt.datetime(counter[i].year, counter[i].month, counter[i].day)].values
            bc_h_line.set_data(x, y)

            # BC Q point (ax2)
            x = ((counter[i] - dt.datetime(counter[i].year, counter[i].month, counter[i].day)).seconds / 60) + 1
            y = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            ax2.scatter(x, Sub.QBCmin[x][y])

            # BC h point (ax3)
            ax3.scatter(x, Sub.HBCmin[x][y])

            return q_line, wl_line, bc_q_line, bc_h_line, ax2.scatter(x, Sub.QBCmin[x][y], s=150), ax3.scatter(x,
                                                                                                               Sub.HBCmin[
                                                                                                                   x][
                                                                                                                   y],
                                                                                                               s=150), day_text

        # plt.tight_layout()

        anim = animation.FuncAnimation(fig2, animate_min, init_func=init_min, frames=len(Sub.q.index),
                                       interval=interval, blit=True)
        return anim

    
    def river1d(self, Sub, start, end, interval=0.00002, xs=0,
                xsbefore=10, xsafter=10, fmt="%Y-%m-%d", textlocation=2,
                xaxislabelsize=15, yaxislabelsize=15, nxlabels=50, plotbanhfuldepth=False):
        """


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

        start = dt.datetime.strptime(start, fmt)
        # end = dt.datetime.strptime(end, fmt) - dt.timedelta(minutes=int(Sub.dt / 60))
        end = dt.datetime.strptime(end, fmt)

        assert start in Sub.referenceindex_results, (
                "plot start date in not in the 1min results, the results starts from " + str(
            Sub.referenceindex_results[0]) + " - and ends on " + str(Sub.referenceindex_results[-1]))
        assert end in Sub.referenceindex_results, ("plot end date in not in the 1min results, the results starts from " + str(
            Sub.referenceindex_results[0]) + " - and ends on " + str(Sub.referenceindex_results[-1]))

        counter = Sub.referenceindex_results[np.where(Sub.referenceindex_results == start)[0][0]:np.where(Sub.referenceindex_results == end)[0][0]+1]
#%%
        margin = 10
        fig2 = plt.figure(20, figsize=(20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=6, figure=fig2)

        # USBC
        ax1 = fig2.add_subplot(gs[0, 0])
        # ax1.set_xlim(1, 1440)
        ax1.set_ylim(0, int(Sub.usbc.max() + margin))
        ax1.set_xlabel('Time', fontsize=15)
        if Sub.usbc.columns[0] == 'q':
            # ax1.set_ylabel('USBC - Q (m3/s)', fontsize=15)
            ax1.set_title("USBC - Q (m3/s)", fontsize=20)
        else:
            # ax1.set_ylabel('USBC - H (m)', fontsize=15)
            ax1.set_title("USBC - H (m)", fontsize=20)
        # ax1.legend(["Q"], fontsize=10)
        ax1.set_xlim(1, 25)
        usbc_line, = ax1.plot([], [], linewidth=5)
        usbc_point = ax1.scatter([], [], s=150)
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

        ax2.set_ylim(np.nanmin(Sub.q)-10, int(np.nanmax(Sub.q))+10) #Sub.q.max().max()
        ax2.tick_params(labelsize=xaxislabelsize)
        ax2.locator_params(axis="x", nbins=nxlabels)
        ax2.set_xlabel('Cross section No', fontsize=xaxislabelsize)
        ax2.set_title("Discharge (m3/s)", fontsize=20)
        # ax2.set_ylabel('Discharge (m3/s)', fontsize=yaxislabelsize, labelpad=0.5)
        ax2.legend(["Discharge"], fontsize=15)

        q_line, = ax2.plot([], [], linewidth=5)
        ax2.grid()

        ### BC

        # DSBC
        ax3 = fig2.add_subplot(gs[0, 5:6])
        ax3.set_xlim(1, 25)
        ax3.set_ylim(0, float(Sub.dsbc.min() + margin))

        ax3.set_xlabel('Time', fontsize=15)
        if Sub.dsbc.columns[0] == 'q':
            # ax3.set_ylabel('DSBC', fontsize=15, labelpad=0.5)
            ax3.set_title("DSBC - Q (m3/s)", fontsize=20)
        else:
            # ax3.set_ylabel('USBC', fontsize=15, labelpad=0.5)
            ax3.set_title("DSBC - H(m)", fontsize=20)

        # ax3.legend(["WL"], fontsize=10)

        dsbc_line, = ax3.plot([], [], linewidth=5)
        # dsbc_point = ax3.scatter([], [], s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig2.add_subplot(gs[1, 0:6])

        if xs == 0:
            ax4.set_xlim(Sub.xsname[0] - 1, Sub.xsname[-1] + 1)
            ax4.set_xticks(Sub.xsname)
            ymin = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'bed level'].values.min()
            ymax = Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'bed level'].values.max()
            ax4.set_ylim(ymin, ymax + np.nanmax(Sub.h) + 5) #Sub.h.max().max()
        else:
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureFirstXS, 'bed level'].values,
                         Sub.crosssections.loc[Sub.crosssections['xsid'] == FigureLastXS, 'zr'].values + 5)

        ax4.tick_params(labelsize=xaxislabelsize)
        ax4.locator_params(axis="x", nbins=nxlabels)

        ax4.plot(Sub.xsname, Sub.crosssections['bed level'], 'k-', linewidth=5, label='Ground level')
        if plotbanhfuldepth :
            ax4.plot(Sub.xsname, Sub.crosssections['bed level'] + Sub.crosssections['depth'], 'k',
                     linewidth=2, label='Bankful depth')

        ax4.set_title("Water surface Profile Simulation", fontsize=15)
        ax4.legend(['Ground level', 'Bankful depth'], fontsize=10)
        ax4.set_xlabel("Profile", fontsize=10)
        ax4.set_ylabel("Elevation m", fontsize=10)
        ax4.grid()

        if xs == 0:
            textlocation = textlocation + Sub.xsname[0]
            day_text = ax4.annotate(' ', xy=(textlocation, Sub.crosssections['bed level'].min()+1),
                                    fontsize=20)
        else:
            day_text = ax4.annotate(' ', xy=(FigureFirstXS + textlocation,
                                                    Sub.crosssections.loc[
                                                        Sub.crosssections['xsid'] == FigureLastXS, 'gl'].values + 1),
                                    fontsize=20)

        wl_line, = ax4.plot([], [], linewidth=5)
        # hLline, = ax4.plot([], [], linewidth=5)

        gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)
        # animation
        plt.show()
#%%
        # animation
        def init_min():
            q_line.set_data([], [])
            wl_line.set_data([], [])
            day_text.set_text('')
            usbc_line.set_data([], [])
            dsbc_line.set_data([], [])
            # bc_q_point
            return q_line, wl_line, day_text, usbc_line, dsbc_line #usbc_point,dsbc_point,

        # animation function. this is called sequentially
        def animate_min(i):
            day_text.set_text('Date = ' + str(counter[i]))

            # discharge (ax1)
            x = Sub.xsname
            # y = Sub.q[Sub.q.index == counter[i]].values[0]
            y = Sub.q[np.where(Sub.referenceindex_results == counter[i])[0][0],:]
            q_line.set_data(x, y)

            # water level (ax4)
            # y = Sub.h.loc[Sub.q.index == counter[i]].values[0]
            y = Sub.wl[np.where(Sub.referenceindex_results == counter[i])[0][0],:]
            wl_line.set_data(x, y)

            # USBC
            f = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            y = Sub.usbc.loc[f:f + dt.timedelta(days=1)].resample('H').mean().interpolate('linear').values
            x = hours[:len(y)]
            usbc_line.set_data(x, y)

            # DSBC
            y = Sub.dsbc.loc[f:f + dt.timedelta(days=1)].resample('H').mean().interpolate('linear').values
            dsbc_line.set_data(x, y)

            # x = counter[i][1]
            # y = Sub.referenceindex.loc[counter[i][0], 'date']
            # ax2.scatter(x, Sub.QBC[x][y])

            # # BC Q point (ax2)
            # x = ((counter[i] - dt.datetime(counter[i].year, counter[i].month, counter[i].day)).seconds / 60) + 1
            # y = dt.datetime(counter[i].year, counter[i].month, counter[i].day)
            # ax2.scatter(x, Sub.USBC[x][y])

            return q_line, wl_line, day_text, usbc_line, dsbc_line #ax2.scatter(x, Sub.USBC[x][y], s=150),

        # plt.tight_layout()

        anim = animation.FuncAnimation(fig2, animate_min, init_func=init_min, frames=len(counter),
                                       interval=interval, blit=True)
        #%%
        return anim


    @staticmethod
    def SaveProfileAnimation(Anim, Path='', fps=3, ffmpegPath=''):

        message = """
        please visit https://ffmpeg.org/ and download a version of ffmpeg compitable
        with your operating system, and copy the content of the folder and paste it
        in the "c:/user/.matplotlib/ffmpeg-static/"
        """
        if ffmpegPath == '':
            ffmpegPath = os.getenv("HOME") + "/.matplotlib/ffmpeg-static/bin/ffmpeg.exe"
            assert os.path.exists(ffmpegPath), message

        mpl.rcParams['animation.ffmpeg_path'] = ffmpegPath

        Ext = Path.split('.')[1]
        if Ext == "gif":
            assert len(Path) >= 1 and Path.endswith(".gif"), "please enter a valid path to save the animation"

            writergif = animation.PillowWriter(fps=fps)
            Anim.save(Path, writer=writergif)
        else:
            try:
                if Ext == 'avi' or Ext == 'mov':
                    writervideo = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    Anim.save(Path, writer=writervideo)
                elif Ext == 'mp4':
                    writermp4 = animation.FFMpegWriter(fps=fps, bitrate=1800)
                    Anim.save(Path, writer=writermp4)

            except FileNotFoundError:
                print(
                    "please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system, for more details please check the method definition")

    def CrossSections(self, Sub):
        """

        plot all the cross sections of the sub-basin

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object.

        Returns
        -------
        None.

        """
        # names = ['gl','zl','zr','hl','hr','bl','br','b','dbf']
        names = list(range(1, 17))
        XSS = pd.DataFrame(columns=names, index=Sub.crosssections['xsid'].values)
        # xsname = Sub.crosssections['xsid'].tolist()

        for i in range(len(Sub.crosssections.index)):

            XSS[1].loc[XSS.index == XSS.index[i]] = 0
            XSS[2].loc[XSS.index == XSS.index[i]] = 0
            bl = Sub.crosssections['bl'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            br = Sub.crosssections['br'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]

            XSS[3].loc[XSS.index == XSS.index[i]] = bl
            XSS[4].loc[XSS.index == XSS.index[i]] = bl
            XSS[5].loc[XSS.index == XSS.index[i]] = bl + b
            XSS[6].loc[XSS.index == XSS.index[i]] = bl + b
            XSS[7].loc[XSS.index == XSS.index[i]] = bl + b + br
            XSS[8].loc[XSS.index == XSS.index[i]] = bl + b + br

            gl = Sub.crosssections['gl'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            subtract = gl
            #    subtract = 0

            zl = Sub.crosssections['zl'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            zr = Sub.crosssections['zr'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]

            hl = Sub.crosssections['hl'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            hr = Sub.crosssections['hr'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]

            XSS[9].loc[XSS.index == XSS.index[i]] = zl - subtract
            if Sub.Version == 1:
                XSS[10].loc[XSS.index == XSS.index[i]] = gl + hl - subtract
                XSS[11].loc[XSS.index == XSS.index[i]] = gl - subtract
                XSS[14].loc[XSS.index == XSS.index[i]] = gl - subtract
                XSS[15].loc[XSS.index == XSS.index[i]] = gl + hr - subtract
            else:
                XSS[10].loc[XSS.index == XSS.index[i]] = gl + dbf + hl - subtract
                XSS[11].loc[XSS.index == XSS.index[i]] = gl + dbf - subtract
                XSS[14].loc[XSS.index == XSS.index[i]] = gl + dbf - subtract
                XSS[15].loc[XSS.index == XSS.index[i]] = gl + dbf + hr - subtract

            XSS[12].loc[XSS.index == XSS.index[i]] = gl - subtract
            XSS[13].loc[XSS.index == XSS.index[i]] = gl - subtract

            XSS[16].loc[XSS.index == XSS.index[i]] = zr - subtract

        # to plot cross section where there is -ve discharge
        # XsId = NegXS[0]
        # to plot cross section you want
        xsno = -1
        # len(Sub.xsname)/9
        rows = 3
        columns = 3

        titlesize = 15
        textsize = 15
        for i in range(int(math.ceil(len(Sub.xsname) / 9.0))):
            fig = plt.figure(1000 + i, figsize=(18, 10))
            gs = gridspec.GridSpec(rows, columns)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS1 = fig.add_subplot(gs[0, 0])
            ax_XS1.plot(xcoord, ycoord, linewidth=6)
            ax_XS1.title.set_text('xs ID = ' + str(XsId))
            ax_XS1.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS1.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS1.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS1.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS2 = fig.add_subplot(gs[0, 1])
            ax_XS2.plot(xcoord, ycoord, linewidth=6)
            ax_XS2.title.set_text('xs ID = ' + str(XsId))
            ax_XS2.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS2.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS2.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS2.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS3 = fig.add_subplot(gs[0, 2])
            ax_XS3.plot(xcoord, ycoord, linewidth=6)
            ax_XS3.title.set_text('xs ID = ' + str(XsId))
            ax_XS3.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS3.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS3.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS3.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS4 = fig.add_subplot(gs[1, 0])
            ax_XS4.plot(xcoord, ycoord, linewidth=6)
            ax_XS4.title.set_text('xs ID = ' + str(XsId))
            ax_XS4.title.set_fontsize(titlesize)
            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS4.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS4.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS4.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS5 = fig.add_subplot(gs[1, 1])
            ax_XS5.plot(xcoord, ycoord, linewidth=6)
            ax_XS5.title.set_text('xs ID = ' + str(XsId))
            ax_XS5.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS5.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS5.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS5.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS6 = fig.add_subplot(gs[1, 2])
            ax_XS6.plot(xcoord, ycoord, linewidth=6)
            ax_XS6.title.set_text('xs ID = ' + str(XsId))
            ax_XS6.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS6.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS6.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS6.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS7 = fig.add_subplot(gs[2, 0])
            ax_XS7.plot(xcoord, ycoord, linewidth=6)
            ax_XS7.title.set_text('xs ID = ' + str(XsId))
            ax_XS7.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS7.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS7.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS7.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS8 = fig.add_subplot(gs[2, 1])
            ax_XS8.plot(xcoord, ycoord, linewidth=6)
            ax_XS8.title.set_text('xs ID = ' + str(XsId))
            ax_XS8.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS8.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS8.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS8.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            xsno = xsno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[xsno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[xsno]].values.tolist()[0]
            b = Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            bl = Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
            ax_XS9 = fig.add_subplot(gs[2, 2])
            ax_XS9.plot(xcoord, ycoord, linewidth=6)
            ax_XS9.title.set_text('xs ID = ' + str(XsId))
            ax_XS9.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf = Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[xsno]].values[0]
                ax_XS9.annotate("dbf=" + str(round(dbf, 2)), xy=(bl, dbf - 0.5), fontsize=textsize)
                # ax_XS9.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS9.annotate("b=" + str(round(b, 2)), xy=(bl, 0), fontsize=textsize)

            gs.update(wspace=0.2, hspace=0.2, top=0.96, bottom=0.1, left=0.05, right=0.96)


    def Plot1minProfile(self, Sub, date, xaxislabelsize=10, nxlabels=50, fmt="%Y-%m-%d"):

        date = dt.datetime.strptime(date, fmt)

        fig50, ax1 = plt.subplots(figsize=(15, 8))
        ax2 = ax1.twinx()
        ax1.plot(Sub.Qmin.columns, Sub.Qmin[Sub.Qmin.index == date].values[0], 'r')
        ax2.plot(Sub.Hmin.columns, Sub.Hmin[Sub.Qmin.index == date].values[0])  # -Sub.crosssections['gl'].values
        ax1.set_ylabel('Discharge', fontsize=20)
        ax2.set_ylabel('Water level', fontsize=20)
        ax1.set_xlabel('Cross sections', fontsize=20)
        ax1.set_xticks(Sub.xsname)
        ax1.tick_params(labelsize=xaxislabelsize)
        ax1.locator_params(axis="x", nbins=nxlabels)

        ax1.grid()

    def AnimateArray(Arr, Time, NoElem, TicksSpacing=2, Figsize=(8, 8), PlotNumbers=True,
                     NumSize=8, Title='Total Discharge', titlesize=15, Backgroundcolorthreshold=None,
                     cbarlabel='Discharge m3/s', cbarlabelsize=12, textcolors=("white", "black"),
                     Cbarlength=0.75, interval=200, cmap='coolwarm_r', Textloc=[0.1, 0.2],
                     Gaugecolor='red', Gaugesize=100, ColorScale=1, gamma=1. / 2., linthresh=0.0001,
                     linscale=0.001, midpoint=0, orientation='vertical', rotation=-90, IDcolor="blue",
                     IDsize=10, **kwargs):
        """

        Parameters
        ----------
        Arr : [array]
            the array you want to animate.
        Time : [dataframe]
            dataframe contains the date of values.
        NoElem : [integer]
            Number of the cells that has values.
        TicksSpacing : [integer], optional
            Spacing in the colorbar ticks. The default is 2.
        Figsize : [tuple], optional
            figure size. The default is (8,8).
        PlotNumbers : [bool], optional
            True to plot the values intop of each cell. The default is True.
        NumSize : integer, optional
            size of the numbers plotted intop of each cells. The default is 8.
        Title : [str], optional
            title of the plot. The default is 'Total Discharge'.
        titlesize : [integer], optional
            title size. The default is 15.
        Backgroundcolorthreshold : [float/integer], optional
            threshold value if the value of the cell is greater, the plotted
            numbers will be black and if smaller the plotted number will be white
            if None given the maxvalue/2 will be considered. The default is None.
        textcolors : TYPE, optional
            Two colors to be used to plot the values i top of each cell. The default is ("white","black").
        cbarlabel : str, optional
            label of the color bar. The default is 'Discharge m3/s'.
        cbarlabelsize : integer, optional
            size of the color bar label. The default is 12.
        Cbarlength : [float], optional
            ratio to control the height of the colorbar. The default is 0.75.
        interval : [integer], optional
            number to controlthe speed of the animation. The default is 200.
        cmap : [str], optional
            color style. The default is 'coolwarm_r'.
        Textloc : [list], optional
            location of the date text. The default is [0.1,0.2].
        Gaugecolor : [str], optional
            color of the points. The default is 'red'.
        Gaugesize : [integer], optional
            size of the points. The default is 100.
        IDcolor : [str]
            the ID of the Point.The default is "blue".
        IDsize : [integer]
            size of the ID text. The default is 10.
        ColorScale : integer, optional
            there are 5 options to change the scale of the colors. The default is 1.
            1- ColorScale 1 is the normal scale
            2- ColorScale 2 is the power scale
            3- ColorScale 3 is the SymLogNorm scale
            4- ColorScale 4 is the PowerNorm scale
            5- ColorScale 5 is the BoundaryNorm scale
            ------------------------------------------------------------------
            gamma : [float], optional
                value needed for option 2 . The default is 1./2..
            linthresh : [float], optional
                value needed for option 3. The default is 0.0001.
            linscale : [float], optional
                value needed for option 3. The default is 0.001.
            midpoint : [float], optional
                value needed for option 5. The default is 0.
            ------------------------------------------------------------------
        orientation : [string], optional
            orintation of the colorbar horizontal/vertical. The default is 'vertical'.
        rotation : [number], optional
            rotation of the colorbar label. The default is -90.
        **kwargs : [dict]
            keys:
                Points : [dataframe].
                    dataframe contains two columns 'cell_row', and cell_col to
                    plot the point at this location

        Returns
        -------
        animation.FuncAnimation.

        """

        fig = plt.figure(60, figsize=Figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        ax = fig.add_subplot(gs[:, :])
        ticks = np.arange(np.nanmin(Arr), np.nanmax(Arr), TicksSpacing)

        if ColorScale == 1:
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, vmin=np.nanmin(Arr), vmax=np.nanmax(Arr), )
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 2:
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, norm=colors.PowerNorm(gamma=gamma,
                                                                           vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)))
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 3:
            linthresh = 1
            linscale = 2
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh,
                                                                            linscale=linscale, base=np.e,
                                                                            vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)))
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif ColorScale == 4:
            bounds = np.arange(np.nanmin(Arr), np.nanmax(Arr), TicksSpacing)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, norm=norm)
            cbar_kw = dict(ticks=ticks)
        else:
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, norm=MidpointNormalize(midpoint=midpoint))
            cbar_kw = dict(ticks=ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=Cbarlength, orientation=orientation, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=rotation, va="bottom")
        cbar.ax.tick_params(labelsize=10)

        day_text = ax.text(Textloc[0], Textloc[1], ' ', fontsize=cbarlabelsize)
        ax.set_title(Title, fontsize=titlesize)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])
        Indexlist = list()

        for x in range(Arr.shape[0]):
            for y in range(Arr.shape[1]):
                if not np.isnan(Arr[x, y, 0]):
                    Indexlist.append([x, y])

        Textlist = list()
        for x in range(NoElem):
            Textlist.append(ax.text(Indexlist[x][1], Indexlist[x][0],
                                    round(Arr[Indexlist[x][0], Indexlist[x][1], 0], 2),
                                    ha="center", va="center", color="w", fontsize=NumSize))
        # Points = list()
        PoitsID = list()
        if 'Points' in kwargs.keys():
            row = kwargs['Points'].loc[:, 'cell_row'].tolist()
            col = kwargs['Points'].loc[:, 'cell_col'].tolist()
            IDs = kwargs['Points'].loc[:, 'id'].tolist()
            Points = ax.scatter(col, row, color=Gaugecolor, s=Gaugesize)

            for i in range(len(row)):
                PoitsID.append(ax.text(col[i], row[i], IDs[i], ha="center",
                                       va="center", color=IDcolor, fontsize=IDsize))

        # Normalize the threshold to the images color range.
        if Backgroundcolorthreshold is not None:
            Backgroundcolorthreshold = im.norm(Backgroundcolorthreshold)
        else:
            Backgroundcolorthreshold = im.norm(np.nanmax(Arr)) / 2.

        def init():
            im.set_data(Arr[:, :, 0])
            day_text.set_text('')

            output = [im, day_text]

            if 'Points' in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs['Points'].loc[:, 'cell_row'].tolist()
                col = kwargs['Points'].loc[:, 'cell_col'].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], 0], 2)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output

        def animate(i):
            im.set_data(Arr[:, :, i])
            day_text.set_text('Date = ' + str(Time[i])[0:10])

            output = [im, day_text]

            if 'Points' in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs['Points'].loc[:, 'cell_row'].tolist()
                col = kwargs['Points'].loc[:, 'cell_col'].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], i], 2)
                    kw = dict(color=textcolors[
                        int(im.norm(Arr[Indexlist[x][0], Indexlist[x][1], i]) > Backgroundcolorthreshold)])
                    Textlist[x].update(kw)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output

        plt.tight_layout()
        # global anim
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.shape(Arr)[2],
                                       interval=interval, blit=True)

        return anim

    def SaveAnimation(anim, VideoFormat="gif", Path='', SaveFrames=20):
        """


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

        mpl.rcParams['animation.ffmpeg_path'] = ffmpegPath

        if VideoFormat == "gif":
            writergif = animation.PillowWriter(fps=SaveFrames)
            anim.save(Path, writer=writergif)
        else:
            try:
                if VideoFormat == 'avi' or VideoFormat == 'mov':
                    writervideo = animation.FFMpegWriter(fps=SaveFrames, bitrate=1800)
                    anim.save(Path, writer=writervideo)
                elif VideoFormat == 'mp4':
                    writermp4 = animation.FFMpegWriter(fps=SaveFrames, bitrate=1800)
                    anim.save(Path, writer=writermp4)
            except FileNotFoundError:
                print(
                    "please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system, for more details please check the method definition")

    def Plot_Type1(Y1, Y2, Points, PointsY, PointMaxSize=200,
                   PointMinSize=1, X_axis_label='X Axis', LegendNum=5, LegendLoc=(1.3, 1),
                   PointLegendTitle="Output 2", Ylim=[0, 180], Y2lim=[-2, 14],
                   color1='#27408B', color2='#DC143C', color3="grey",
                   linewidth=4, **kwargs):
        """


        Parameters
        ----------
        Y1 : TYPE
            DESCRIPTION.
        Y2 : TYPE
            DESCRIPTION.
        Points : TYPE
            DESCRIPTION.
        PointsY : TYPE
            DESCRIPTION.
        PointMaxSize : TYPE, optional
            DESCRIPTION. The default is 200.
        PointMinSize : TYPE, optional
            DESCRIPTION. The default is 1.
        X_axis_label : TYPE, optional
            DESCRIPTION. The default is 'X Axis'.
        LegendNum : TYPE, optional
            DESCRIPTION. The default is 5.
        LegendLoc : TYPE, optional
            DESCRIPTION. The default is (1.3, 1).
        PointLegendTitle : TYPE, optional
            DESCRIPTION. The default is "Output 2".
        Ylim : TYPE, optional
            DESCRIPTION. The default is [0,180].
        Y2lim : TYPE, optional
            DESCRIPTION. The default is [-2,14].
        color1 : TYPE, optional
            DESCRIPTION. The default is '#27408B'.
        color2 : TYPE, optional
            DESCRIPTION. The default is '#DC143C'.
        color3 : TYPE, optional
            DESCRIPTION. The default is "grey".
        linewidth : TYPE, optional
            DESCRIPTION. The default is 4.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ax1 : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        fig : TYPE
            DESCRIPTION.

        """

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        ax2 = ax1.twinx()

        ax1.plot(Y1[:, 0], Y1[:, 1], zorder=1, color=color1,
                 linestyle=Visualize.LineStyle(0), linewidth=linewidth,
                 label="Model 1 Output1")

        if 'Y1_2' in kwargs.keys():
            Y1_2 = kwargs['Y1_2']

            rows_axis1, cols_axis1 = np.shape(Y1_2)

            if 'Y1_2_label' in kwargs.keys():
                label = kwargs['Y2_2_label']
            else:
                label = ['label'] * (cols_axis1 - 1)
            # first column is the x axis
            for i in range(1, cols_axis1):
                ax1.plot(Y1_2[:, 0], Y1_2[:, i], zorder=1, color=color2,
                         linestyle=Visualize.LineStyle(i), linewidth=linewidth,
                         label=label[i - 1])

        ax2.plot(Y2[:, 0], Y2[:, 1], zorder=1, color=color3,
                 linestyle=Visualize.LineStyle(6), linewidth=2,
                 label="Output1-Diff")

        if 'Y2_2' in kwargs.keys():
            Y2_2 = kwargs['Y2_2']
            rows_axis2, cols_axis2 = np.shape(Y2_2)

            if 'Y2_2_label' in kwargs.keys():
                label = kwargs['Y2_2_label']
            else:
                label = ['label'] * (cols_axis2 - 1)

            for i in range(1, cols_axis2):
                ax1.plot(Y2_2[:, 0], Y2_2[:, i], zorder=1, color=color2,
                         linestyle=Visualize.LineStyle(i), linewidth=linewidth,
                         label=label[i - 1])

        if 'Points1' in kwargs.keys():
            # first axis in the x axis
            Points1 = kwargs['Points1']

            vmax = np.max(Points1[:, 1:])
            vmin = np.min(Points1[:, 1:])

            vmax = max(Points[:, 1].max(), vmax)
            vmin = min(Points[:, 1].min(), vmin)

        else:
            vmax = max(Points)
            vmin = min(Points)

        vmaxnew = PointMaxSize
        vminnew = PointMinSize

        Points_scaled = [ST.Rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points[:, 1]]
        f1 = np.ones(shape=(len(Points))) * PointsY
        scatter = ax2.scatter(Points[:, 0], f1, zorder=1, c=color1,
                              s=Points_scaled, label="Model 1 Output 2")

        if 'Points1' in kwargs.keys():
            row_points, col_points = np.shape(Points1)
            PointsY1 = kwargs['PointsY1']
            f2 = np.ones_like(Points1[:, 1:])

            for i in range(col_points - 1):
                Points1_scaled = [ST.Rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points1[:, i]]
                f2[:, i] = PointsY1[i]

                ax2.scatter(Points1[:, 0], f2[:, i], zorder=1, c=color2,
                            s=Points1_scaled, label="Model 2 Output 2")

        # produce a legend with the unique colors from the scatter
        legend1 = ax2.legend(*scatter.legend_elements(),
                             bbox_to_anchor=(1.1, 0.2))  # loc="lower right", title="RIM"

        ax2.add_artist(legend1)

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=LegendNum)
        # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
        L = [float(i[14:-2]) for i in labels]
        labels1 = [round(ST.Rescale(x, vminnew, vmaxnew, vmin, vmax) / 1000) for x in L]

        legend2 = ax2.legend(handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle)
        ax2.add_artist(legend2)

        ax1.set_ylim(Ylim)
        ax2.set_ylim(Y2lim)
        #
        ax1.set_ylabel('Output 1 (m)', fontsize=12)
        ax2.set_ylabel('Output 1 - Diff (m)', fontsize=12)
        ax1.set_xlabel(X_axis_label, fontsize=12)
        ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
        ax1.tick_params(which='minor', length=5)
        fig.legend(loc="lower center", bbox_to_anchor=(1.3, 0.3), bbox_transform=ax1.transAxes, fontsize=10)
        plt.rcParams.update({'ytick.major.size': 3.5})
        plt.rcParams.update({'font.size': 12})
        plt.title("Model Output Comparison", fontsize=15)

        plt.subplots_adjust(right=0.7)
        # plt.tight_layout()

        return (ax1, ax2), fig

    def ListAttributes(self):
        """
        Print Attributes List
        """

        print('\n')
        print('Attributes List of: ' + repr(self.__dict__['name']) + ' - ' + self.__class__.__name__ + ' Instance\n')
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != 'name':
                print(str(key) + ' : ' + repr(self.__dict__[key]))

        print('\n')


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]

        return np.ma.masked_array(np.interp(value, x, y))
