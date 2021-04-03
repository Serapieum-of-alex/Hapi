"""
Created on Sat Mar 14 16:36:01 2020

@author: mofarrag
"""
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
linestyles = OrderedDict( [('solid', (0, ())),                              #0
                               ('loosely dotted', (0, (1, 10))),                #1
                               ('dotted', (0, (1, 5))),                         #2
                               ('densely dotted', (0, (1, 1))),                 #3
                               ('loosely dashed', (0, (5, 10))),                #4
                               ('dashed',(0, (5, 5))),                          #5
                               ('densely dashed', (0, (5, 1))),                 #6
                               ('loosely dashdotted', (0, (3, 10, 1, 10))),     #7
                               ('dashdotted', (0, (3, 5, 1, 5))),               #8
                               ('densely dashdotted',  (0, (3, 1, 1, 1))),      #9                                      
                               ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))), #10
                               ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),            #11
                               ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),    #12
                               ('densely dashdotdottededited', (0, (6, 1, 1, 1, 1, 1)))])  #13

MarkerStyle = ['--o',':D','-.H','--x',':v','--|','-+','-^','--s','-.*','-.h']

FigureDefaultOptions = dict( ylabel = '', xlabel = '',
                      legend = '', legend_size = 10, figsize = (10,8),
                      labelsize = 10, fontsize = 10, name = 'hist.tif',
                      color1 = '#3D59AB', color2 = "#DC143C", linewidth = 3,
                      Axisfontsize = 15
                      )

hours = list(range(1,25))

class Visualize():

    def __init__(self, Sub, resolution = "Hourly"):
        self.resolution = "Hourly"
        # self.XSname = Sub.crosssections['xsid'].tolist()
    
    @staticmethod
    def LineStyle(Style = 'loosely dotted'):
                
        if type(Style) == str:
            try:
                return linestyles[Style]
            except KeyError:
                print("The Style name you entered-" + Style + "-does not exist please choose from the available styles")
                print(list(linestyles))
        else:
            return list(linestyles.items())[Style][1]
            
    @staticmethod
    def MarkerStyle(Style):
        if Style > len(MarkerStyle)-1:
           Style = Style % len(MarkerStyle)
        return MarkerStyle[Style]
    
    def GroundSurface(self, Sub, XSID='', XSbefore = 10, XSafter = 10, FloodPlain = False):
        
        if XSID == '':
            XS = 0
        else:
            XS = np.where(Sub.crosssections['xsid'] == XSID )[0][0]
        
        GroundSurfacefig = plt.figure(70, figsize = (20,10) )
        gs = gridspec.GridSpec(nrows = 2, ncols = 6, figure = GroundSurfacefig )
        axGS = GroundSurfacefig.add_subplot(gs[0:2,0:6])


        if XS == 0 :
            axGS.set_xlim(Sub.XSname[0]-1,Sub.XSname[-1]+1)
            axGS.set_xticks(Sub.XSname)
        else :
             # not the whole sub-basin
            FigureFirstXS = Sub.XSname[XS] - XSbefore
            FigureLastXS = Sub.XSname[XS] + XSafter
            axGS.set_xlim(FigureFirstXS,FigureLastXS)

            axGS.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            axGS.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        #ax4 = fig.add_subplot(gs[0:2,0:6])


        axGS.tick_params(labelsize= 8)
        # plot dikes
        axGS.plot(Sub.XSname, Sub.crosssections['zl'],'k--', dashes = (5,1), linewidth = 2, label = 'Left Dike')
        axGS.plot(Sub.XSname, Sub.crosssections['zr'],'k.-', linewidth = 2, label = 'Right Dike')
        
        if FloodPlain:
            fpl = Sub.crosssections['gl'] + Sub.crosssections['dbf'] + Sub.crosssections['hl']
            fpr = Sub.crosssections['gl'] + Sub.crosssections['dbf'] + Sub.crosssections['hr']
            axGS.plot(Sub.XSname, fpl,'b-.',linewidth = 2, label = 'Floodplain left')
            axGS.plot(Sub.XSname, fpr,'r-.',linewidth = 2, label = 'Floodplain right')
        
        # plot the bedlevel/baklevel
        if Sub.Version == 1:
            axGS.plot(Sub.XSname, Sub.crosssections['gl'],'k-', linewidth = 5, label = 'Bankful level')
        else:
            axGS.plot(Sub.XSname, Sub.crosssections['gl'],'k-', linewidth = 5, label = 'Ground level')
            axGS.plot(Sub.XSname, Sub.crosssections['gl'] + Sub.crosssections['dbf'],'k',linewidth = 2, label = 'Bankful depth')

        axGS.set_title("Water surface Profile Simulation SubID = " + str(Sub.ID), fontsize = 15)
        axGS.legend(fontsize = 15)
        axGS.set_xlabel("Profile", fontsize = 15)
        axGS.set_ylabel("Elevation m", fontsize = 15)
        axGS.grid()

        # if XS == 0 :
            # day_text = axGS.annotate('',xy=(Sub.XSname[0],Sub.crosssections['gl'].min()),fontsize= 20)
        # else:
            # day_text = axGS.annotate('',xy=(FigureFirstXS+1,Sub.crosssections['gl'][FigureLastXS]+1),fontsize= 20)

        GroundSurfacefig.tight_layout()




    def WaterSurfaceProfile(self, Sub, PlotStart, PlotEnd, Interval = 200, XS=0,
                            XSbefore = 10, XSafter = 10, Save=False, Path = '',
                            SaveFrames=60):
        """
        =============================================================================
            WaterSurfaceProfile(Sub, PlotStart, PlotEnd, interval = 200, XS=0,
                                XSbefore = 10, XSafter = 10)
        =============================================================================

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object.
        PlotStart : [datetime object]
            starting date of the simulation.
        PlotEnd : [datetime object]
            end date of the simulation.
        Interval : [integer], optional
            speed of the animation. The default is 200.
        XS : [integer], optional
            order of a specific cross section to plot the data animation around it. The default is 0.
        XSbefore : [integer], optional
            number of cross sections to be displayed before the chosen cross section . The default is 10.
        XSafter : [integer], optional
            number of cross sections to be displayed after the chosen cross section . The default is 10.
        Save : [Boolen/string]
            different formats to save the animation 'gif', 'avi', 'mov', 'mp4'.The default is False
        Path : [String]
            Path where you want to save the animation, you have to include the
            extension at the end of the path.
        SaveFrames : [integer]
            numper of frames per second

        in order to save a video using matplotlib you have to download ffmpeg from
        https://ffmpeg.org/ and define this path to matplotlib

        import matplotlib as mpl
        mpl.rcParams['animation.ffmpeg_path'] = "path where you saved the ffmpeg.exe/ffmpeg.exe"

        Returns
        -------
        TYPE

        """
        assert PlotStart < PlotEnd, "start Simulation date should be before the end simulation date "
        if Sub.from_beginning == 1:
            Period = Sub.Daylist[np.where(Sub.ReferenceIndex == PlotStart)[0][0]:np.where(Sub.ReferenceIndex == PlotEnd)[0][0]+1]
        else:
            ii = Sub.ReferenceIndex.index[np.where(Sub.ReferenceIndex == PlotStart)[0][0]]
            ii2 = Sub.ReferenceIndex.index[np.where(Sub.ReferenceIndex == PlotEnd)[0][0]]
            Period = list(range(ii,ii2+1))

        counter = [(i,j) for i in Period for j in hours]

        fig = plt.figure(60, figsize = (20,10) )
        gs = gridspec.GridSpec(nrows = 2, ncols = 6, figure = fig )
        ax1 = fig.add_subplot(gs[0,2:6])
        ax1.set_ylim(0,int(Sub.Result1D['q'].max()))



        if XS == 0 :
            # plot the whole sub-basin
            ax1.set_xlim(Sub.XSname[0]-1,Sub.XSname[-1]+1)
            ax1.set_xticks(Sub.XSname)
            ax1.set_xticklabels(Sub.XSname)
        else :
            # not the whole sub-basin
            FigureFirstXS = Sub.XSname[XS] - XSbefore
            FigureLastXS = Sub.XSname[XS] + XSafter
            ax1.set_xlim(FigureFirstXS,FigureLastXS)

            ax1.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax1.set_xticklabels(list(range(FigureFirstXS, FigureLastXS)))

        ax1.tick_params(labelsize= 6)
        ax1.set_xlabel('Cross section No', fontsize= 15)
        ax1.set_ylabel('Discharge (m3/s)', fontsize= 15, labelpad= 0.5)
        ax1.set_title('Sub-Basin'+' ' + str(Sub.ID),fontsize= 15 )
        ax1.legend(["Discharge"],fontsize = 15 )



        Qline, = ax1.plot([],[],linewidth = 5) #Sub.Result1D['q'][Sub.Result1D['day'] == Sub.Result1D['day'][1]][Sub.Result1D['hour'] == 1]
        ax1.grid()

        ### BC
        # Q
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_xlim(1,25)
        ax2.set_ylim(0,int(Sub.QBC.max().max())+1)


        ax2.set_xlabel('Time', fontsize= 15)
        ax2.set_ylabel('Q (m3/s)', fontsize= 15, labelpad= 0.1)
        #ax2.yaxis.set_label_coords(-0.05, int(BC_q_T.max().max()))
        ax2.set_title("BC - Q",fontsize= 20 )
        ax2.legend(["Q"],fontsize = 15 )


        BC_q_line, = ax2.plot([],[],linewidth = 5)
        BC_q_point = ax2.scatter([],[],s=300)
        ax2.grid()

        # h
        ax3 = fig.add_subplot(gs[0,0])
        ax3.set_xlim(1,25)
        ax3.set_ylim(float(Sub.HBC.min().min()),float(Sub.HBC.max().max()))

        ax3.set_xlabel('Time', fontsize= 15)
        ax3.set_ylabel('water level', fontsize= 15, labelpad= 0.5)
        ax3.set_title("BC - H",fontsize= 20 )
        ax3.legend(["WL"],fontsize = 10 )

        BC_h_line, = ax3.plot([],[],linewidth = 5) #Sub.Result1D['q'][Sub.Result1D['day'] == Sub.Result1D['day'][1]][Sub.Result1D['hour'] == 1]
        BC_h_point = ax3.scatter([],[],s=300)
        ax3.grid()

        # water surface profile
        ax4 = fig.add_subplot(gs[1,0:6])

        if XS == 0 :
            ax4.set_xlim(Sub.XSname[0]-1,Sub.XSname[-1]+1)
            ax4.set_xticks(Sub.XSname)
        else :
            ax4.set_xlim(FigureFirstXS, FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS, FigureLastXS)))
            ax4.set_ylim(Sub.crosssections['gl'][FigureFirstXS],Sub.crosssections['zr'][FigureLastXS]+5)

        #ax4 = fig.add_subplot(gs[0:2,0:6])


        ax4.tick_params(labelsize= 8)
        ax4.plot(Sub.XSname, Sub.crosssections['zl'],'k--', dashes = (5,1), linewidth = 2, label = 'Left Dike')
        ax4.plot(Sub.XSname, Sub.crosssections['zr'],'k.-', linewidth = 2, label = 'Right Dike')

        if Sub.Version == 1:
            ax4.plot(Sub.XSname, Sub.crosssections['gl'],'k-', linewidth = 5, label = 'Bankful level')
        else:
            ax4.plot(Sub.XSname, Sub.crosssections['gl'],'k-', linewidth = 5, label = 'Ground level')
            ax4.plot(Sub.XSname, Sub.crosssections['gl']+Sub.crosssections['dbf'],'k',linewidth = 2, label = 'Bankful depth')



        ax4.set_title("Water surface Profile Simulation", fontsize = 15)
        ax4.legend(fontsize = 15)
        ax4.set_xlabel("Profile", fontsize = 15)
        ax4.set_ylabel("Elevation m", fontsize = 15)
        ax4.grid()

        if XS == 0 :
            day_text = ax4.annotate('Begining',xy=(Sub.XSname[0],Sub.crosssections['gl'].min()),fontsize= 20)
        else:
            day_text = ax4.annotate('Begining',xy=(FigureFirstXS+1,Sub.crosssections['gl'][FigureLastXS]+1),fontsize= 20)


        WLline, = ax4.plot([],[],linewidth = 5)
        hLline, = ax4.plot([],[],linewidth = 5)


        gs.update(wspace = 0.2, hspace = 0.2, top= 0.96, bottom = 0.1, left = 0.05, right = 0.96)
        # animation
        plt.show()
        
        def init_q() :
            Qline.set_data([],[])
            WLline.set_data([],[])
            hLline.set_data([],[])
            day_text.set_text('')

            BC_q_line.set_data([],[])
            BC_h_line.set_data([],[])
            BC_q_point
            BC_h_point

            return Qline, WLline, hLline, day_text, BC_q_line, BC_h_line, BC_q_point, BC_h_point

        # animation function. this is called sequentially
        def animate_q(i):
            x = Sub.XSname
            y= Sub.Result1D['q'][Sub.Result1D['day'] == counter[i][0]][Sub.Result1D['hour'] == counter[i][1]].values

            day = Sub.ReferenceIndex.loc[counter[i][0],'date']


            day_text.set_text('day = '+str(day + dt.timedelta(hours = counter[i][1])) )
            Qline.set_data(x,y)

            y= Sub.Result1D['wl'][Sub.Result1D['day'] == counter[i][0]][Sub.Result1D['hour'] == counter[i][1]].values
            WLline.set_data(x,y)

            y= Sub.Result1D['h'][Sub.Result1D['day'] == counter[i][0]][Sub.Result1D['hour'] == counter[i][1]].values*2 + Sub.crosssections['gl'][Sub.crosssections.index[len(Sub.XSname)-1]]
            hLline.set_data(x,y)

            x = Sub.QBC.columns.values
        #    if XS == 0:
        #        y = BC_q_T.loc[Qobs.index[counter[i][0]-1]].values
        #    else:
            y = Sub.QBC.loc[Sub.ReferenceIndex.loc[counter[i][0],'date']].values
            BC_q_line.set_data(x,y)

            # BC H (ax3)
        #    if XS == 0:
        #        y = BC_h_T.loc[Qobs.index[counter[i][0]-1]].values
        #    else:
            y = Sub.HBC.loc[Sub.ReferenceIndex.loc[counter[i][0],'date']].values

            BC_h_line.set_data(x,y)

            #BC Q point (ax2)
            x = counter[i][1]
        #    if XS == 0:
        #        y= Qobs.index[counter[i][0]-1]
        #    else :
            y= Sub.ReferenceIndex.loc[counter[i][0],'date']
            ax2.scatter(x, Sub.QBC[x][y])

            #BC h point (ax3)
            ax3.scatter(x, Sub.QBC[x][y])


            return Qline, WLline, hLline, day_text, BC_q_line, BC_h_line, ax2.scatter(x, Sub.QBC[x][y],s=300),ax3.scatter(x, Sub.HBC[x][y],s=300)
        # plt.tight_layout()

        
        
        anim = animation.FuncAnimation(fig, animate_q, init_func=init_q, frames = np.shape(counter)[0],
                                       interval = Interval, blit = True)
        
        if Save != False:
            if Save == "gif":
                assert len(Path) >= 1 and Path.endswith(".gif"), "please enter a valid path to save the animation"
                writergif = animation.PillowWriter(fps=SaveFrames)
                anim.save(Path, writer=writergif)
            else:
                try:
                    if Save=='avi' or Save=='mov':
                        writervideo = animation.FFMpegWriter(fps=SaveFrames,bitrate=1800)
                        anim.save(Path, writer=writervideo)
                    elif Save=='mp4':
                        writermp4 = animation.FFMpegWriter(fps=SaveFrames,bitrate=1800)
                        anim.save(Path, writer=writermp4)
                except FileNotFoundError:
                    print("please visit https://ffmpeg.org/ and download a version of ffmpeg compitable with your operating system, for more details please check the method definition")
        
        return anim

    def CrossSections(self, Sub):
        """
        =========================================================
            CrossSections(Sub)
        =========================================================
        plot all the cross sections of the sub-basin

        Parameters
        ----------
        Sub : [Object]
            Sub-object created as a sub class from River object.

        Returns
        -------
        None.

        """
        #names = ['gl','zl','zr','hl','hr','bl','br','b','dbf']
        names = list(range(1,17))
        XSS = pd.DataFrame(columns = names, index = Sub.crosssections['xsid'].values)
        # XSname = Sub.crosssections['xsid'].tolist()

        for i in range(len(Sub.crosssections.index)):

            XSS[1].loc[XSS.index == XSS.index[i]] = 0
            XSS[2].loc[XSS.index == XSS.index[i]] = 0
            bl = Sub.crosssections['bl'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]
            br= Sub.crosssections['br'].loc[Sub.crosssections.index == Sub.crosssections.index[i]].values[0]

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

            XSS[9].loc[XSS.index == XSS.index[i]] = zl-subtract
            if Sub.Version == 1:
                XSS[10].loc[XSS.index == XSS.index[i]] = gl + hl -subtract
                XSS[11].loc[XSS.index == XSS.index[i]] = gl - subtract
                XSS[14].loc[XSS.index == XSS.index[i]] = gl - subtract
                XSS[15].loc[XSS.index == XSS.index[i]] = gl + hr - subtract
            else:
                XSS[10].loc[XSS.index == XSS.index[i]] = gl + dbf + hl -subtract
                XSS[11].loc[XSS.index == XSS.index[i]] = gl + dbf - subtract
                XSS[14].loc[XSS.index == XSS.index[i]] = gl + dbf - subtract
                XSS[15].loc[XSS.index == XSS.index[i]] = gl + dbf + hr - subtract

            XSS[12].loc[XSS.index == XSS.index[i]] = gl - subtract
            XSS[13].loc[XSS.index == XSS.index[i]] = gl - subtract

            XSS[16].loc[XSS.index == XSS.index[i]] = zr - subtract


        # to plot cross section where there is -ve discharge
        #XsId = NegXS[0]
        # to plot cross section you want
        XSno = -1
        #len(Sub.XSname)/9
        rows = 3
        columns = 3

        titlesize = 15
        textsize = 15
        for i in range(int(math.ceil(len(Sub.XSname)/9.0))):
            fig = plt.figure(1000+i, figsize=(18,10))
            gs = gridspec.GridSpec(rows, columns)

            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS1 = fig.add_subplot(gs[0,0])
            ax_XS1.plot(xcoord,ycoord, linewidth = 6)
            ax_XS1.title.set_text('XS ID = '+str(XsId))
            ax_XS1.title.set_fontsize(titlesize )

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS1.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS1.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS1.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS2 = fig.add_subplot(gs[0,1])
            ax_XS2.plot(xcoord,ycoord, linewidth = 6)
            ax_XS2.title.set_text('XS ID = '+str(XsId))
            ax_XS2.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS2.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS2.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS2.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS3 = fig.add_subplot(gs[0,2])
            ax_XS3.plot(xcoord,ycoord, linewidth = 6)
            ax_XS3.title.set_text('XS ID = '+str(XsId))
            ax_XS3.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS3.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS3.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS3.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS4 = fig.add_subplot(gs[1,0])
            ax_XS4.plot(xcoord,ycoord, linewidth = 6)
            ax_XS4.title.set_text('XS ID = '+str(XsId))
            ax_XS4.title.set_fontsize(titlesize)
            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS4.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS4.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS4.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS5 = fig.add_subplot(gs[1,1])
            ax_XS5.plot(xcoord,ycoord, linewidth = 6)
            ax_XS5.title.set_text('XS ID = '+str(XsId))
            ax_XS5.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS5.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS5.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS5.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS6 = fig.add_subplot(gs[1,2])
            ax_XS6.plot(xcoord,ycoord, linewidth = 6)
            ax_XS6.title.set_text('XS ID = '+str(XsId))
            ax_XS6.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS6.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS6.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS6.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )

            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS7 = fig.add_subplot(gs[2,0])
            ax_XS7.plot(xcoord,ycoord, linewidth = 6)
            ax_XS7.title.set_text('XS ID = '+str(XsId))
            ax_XS7.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS7.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS7.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS7.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS8 = fig.add_subplot(gs[2,1])
            ax_XS8.plot(xcoord,ycoord, linewidth = 6)
            ax_XS8.title.set_text('XS ID = '+str(XsId))
            ax_XS8.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS8.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS8.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

            ax_XS8.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize )


            XSno = XSno + 1
            XsId = Sub.crosssections['xsid'][Sub.crosssections.index[XSno]]
            xcoord = XSS[names[0:8]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            ycoord = XSS[names[8:16]].loc[XSS.index == XSS.index[XSno]].values.tolist()[0]
            b= Sub.crosssections['b'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            bl= Sub.crosssections['bl'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
            ax_XS9 = fig.add_subplot(gs[2,2])
            ax_XS9.plot(xcoord,ycoord, linewidth = 6)
            ax_XS9.title.set_text('XS ID = '+str(XsId))
            ax_XS9.title.set_fontsize(titlesize)

            if Sub.Version > 1:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS9.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                # ax_XS9.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
            ax_XS9.annotate("b="+str(round(b,2)),xy=(bl,0), fontsize = textsize  )

            gs.update(wspace = 0.2, hspace = 0.2, top= 0.96, bottom = 0.1, left = 0.05, right = 0.96)


    def WaterSurfaceProfile1Min(self, Sub, PlotStart, PlotEnd, Interval = 0.00002, XS=0,
                            XSbefore = 10, XSafter = 10):


        assert PlotStart in Sub.Hmin.index, ("plot start date in not read")
        assert PlotEnd in Sub.Hmin.index,("plot end date in not read")

        counter = Sub.Hmin.index[np.where(Sub.Hmin.index == PlotStart)[0][0]:np.where(Sub.Hmin.index == PlotEnd)[0][0]]

        fig2 = plt.figure(20, figsize = (20,10) )
        gs = gridspec.GridSpec(nrows = 2, ncols = 6, figure = fig2)

        ax1 = fig2.add_subplot(gs[0,2:6])

        if XS == 0 :
            # plot the whole sub-basin
            ax1.set_xlim(Sub.XSname[0]-1,Sub.XSname[-1]+1)
            ax1.set_xticks(Sub.XSname)
            ax1.set_xticklabels(Sub.XSname)
        else :
            # not the whole sub-basin
            FigureFirstXS = Sub.XSname[0]-1
            FigureLastXS = FigureFirstXS +25
            ax1.set_xlim(FigureFirstXS,FigureLastXS)
            ax1.set_xticks(list(range(FigureFirstXS,FigureLastXS)))
            ax1.set_xticklabels(list(range(FigureFirstXS,FigureLastXS)))

        ax1.set_ylim(0,int(Sub.Result1D['q'].max()))
        if len(Sub.XSname) > 35:
            ax1.tick_params(labelsize= 6)
        else :
            ax1.tick_params(labelsize= 8)

        ax1.set_xlabel('Cross section No', fontsize= 20)
        ax1.set_ylabel('Discharge (m3/s)', fontsize= 20)
        ax1.set_title('Sub-Basin'+' ' + str(Sub.ID),fontsize= 20 )
        #ax1.legend(["Discharge"],fontsize = 10 )

        Qline, = ax1.plot([],[],linewidth = 5) #Sub.Result1D['q'][Sub.Result1D['day'] == Sub.Result1D['day'][1]][Sub.Result1D['hour'] == 1]
        ax1.grid()

        ### BC
        # Q
        ax2 = fig2.add_subplot(gs[0,1:2])
        ax2.set_xlim(1,1440)
        ax2.set_ylim(0,int(Sub.QBC.max().max()))
        #ax2.set_xticks(XSname)
        #ax2.set_xticklabels(XSname)
        #if len(XSname) > 35:
        #    ax2.tick_params(labelsize= 5)
        #else :
        #    ax2.tick_params(labelsize= 8)

        ax2.set_xlabel('Time', fontsize= 15)
        ax2.set_ylabel('Discharge (m3/s)', fontsize= 15)
        ax2.set_title("BC - Q",fontsize= 20 )
        ax2.legend(["Q"],fontsize = 10 )


        BC_q_line, = ax2.plot([],[],linewidth = 5)
        BC_q_point = ax2.scatter([],[],s=150)
        #BC_q_point = ax2.vlines(0,0,int(BC_q.max().max()), colors='k', linestyles='solid',linewidth = 5)
        ax2.grid()

        # h
        ax3 = fig2.add_subplot(gs[0,0:1])
        ax3.set_xlim(1,1440)
        #ax3.set_ylim(Sub.crosssections['gl'][Sub.crosssections['xsid']==XSname[0]].values[0],float(BC_h.max().max()))
        ax3.set_ylim(float(Sub.HBC.min().min()),float(Sub.HBC.max().max()))
        #ax3.set_xticks(XSname)
        #ax3.set_xticklabels(XSname)
        #if len(XSname) > 35:
        #    ax3.tick_params(labelsize= 5)
        #else :
        #    ax3.tick_params(labelsize= 8)

        ax3.set_xlabel('Time', fontsize= 15)
        ax3.set_ylabel('water level', fontsize= 15)
        ax3.set_title("BC - H",fontsize= 20 )
        ax3.legend(["WL"],fontsize = 10 )

        BC_h_line, = ax3.plot([],[],linewidth = 5) #Sub.Result1D['q'][Sub.Result1D['day'] == Sub.Result1D['day'][1]][Sub.Result1D['hour'] == 1]
        BC_h_point = ax3.scatter([],[],s=150)
        #BC_h_point = ax3.vlines(x=[],ymin=[],ymax=[], colors='k', linestyles='solid',linewidth = 5)
        ax3.grid()


        # water surface profile
        ax4 = fig2.add_subplot(gs[1,0:6])
        #ax4 = fig2.add_subplot(gs[0:2,0:6])

        if XS == 0 :
            ax4.set_xlim(Sub.XSname[0]-1,Sub.XSname[-1]+1)
            ax4.set_xticks(Sub.XSname)
        else :
            ax4.set_xlim(FigureFirstXS,FigureLastXS)
            ax4.set_xticks(list(range(FigureFirstXS,FigureLastXS)))
            ax4.set_ylim(Sub.crosssections['gl'][Sub.LastXS],Sub.crosssections['zr'][FigureFirstXS]+5)

        #    ax1.set_xticklabels(list(range(Sub.FirstXS,Sub.LastXS)))



        if len(Sub.XSname) > 30 :
            ax4.tick_params(labelsize= 7)
        else:
            ax4.tick_params(labelsize= 8)

        ax4.plot(Sub.XSname, Sub.crosssections['gl'],'k-', linewidth = 5)
        ax4.plot(Sub.XSname, Sub.crosssections['zl'],'k--', dashes = (5,1), linewidth = 2)
        ax4.plot(Sub.XSname, Sub.crosssections['zr'],'k.-', linewidth = 2)
        ax4.plot(Sub.XSname, Sub.crosssections['gl']+Sub.crosssections['dbf'],'k',linewidth = 2)

        ax4.set_title("Water surface Profile Simulation", fontsize = 15)
        ax4.legend(['Ground level','Left Dike','Right Dike','Bankful depth'],fontsize = 10)
        ax4.set_xlabel("Profile", fontsize = 10)
        ax4.set_ylabel("Elevation m", fontsize = 10)
        ax4.grid()

        if XS == 0 :
            day_text = ax4.annotate('',xy=(Sub.XSname[0],Sub.crosssections['gl'].min()),fontsize= 20)
        else:
            day_text = ax4.annotate('',xy=(FigureFirstXS+1,Sub.crosssections['gl'][FigureLastXS]+1),fontsize= 20)

        WLline, = ax4.plot([],[],linewidth = 5)

        # animation

        def init_min() :
            Qline.set_data([],[])
            WLline.set_data([],[])
            day_text.set_text('')
            BC_q_line.set_data([],[])
            BC_h_line.set_data([],[])
            BC_q_point
            BC_h_point
            return Qline, WLline, BC_q_line, BC_h_line, BC_q_point, BC_h_point, day_text

        # animation function. this is called sequentially
        def animate_min(i):
            # discharge (ax1)
            x = Sub.XSname
            y= Sub.Qmin[Sub.Qmin.index == counter[i]].values[0]
            day_text.set_text('Date = ' + str(counter[i]) )
            Qline.set_data(x,y)
            # water level (ax4)
            y= Sub.Hmin.loc[Sub.Qmin.index == counter[i]].values[0]
            WLline.set_data(x,y)
            # BC Q (ax2)
            x = Sub.QBCmin.columns.values
            y = Sub.QBCmin.loc[dt.datetime(counter[i].year,counter[i].month,counter[i].day)].values
            BC_q_line.set_data(x,y)

            # BC H (ax3)
            y = Sub.HBCmin.loc[dt.datetime(counter[i].year,counter[i].month,counter[i].day)].values
            BC_h_line.set_data(x,y)

            #BC Q point (ax2)
            x=(counter[i] - dt.datetime(counter[i].year,counter[i].month,counter[i].day)).seconds/60
            y= dt.datetime(counter[i].year,counter[i].month,counter[i].day)
            ax2.scatter(x, Sub.QBCmin[x][y])

            #BC h point (ax3)
            ax3.scatter(x, Sub.HBCmin[x][y])

            return Qline, WLline, BC_q_line, BC_h_line, ax2.scatter(x, Sub.QBCmin[x][y],s=150), ax3.scatter(x, Sub.HBCmin[x][y],s=150), day_text

        plt.tight_layout()

        anim = animation.FuncAnimation(fig2, animate_min, init_func=init_min, frames = len(Sub.Qmin.index),
                                       interval = Interval, blit = True)
        return anim
    
    def AnimateArray(Arr, Time, NoElem, TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
                     NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None, 
                     cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
                     Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
                     Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
                     linscale=0.001, midpoint=0, orientation='vertical', rotation=-90, IDcolor = "blue",
                     IDsize =10, **kwargs):
        """
         =============================================================================
           AnimateArray(Arr, Time, NoElem, TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
                  NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None, 
                  cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
                  Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
                  Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
                  linscale=0.001, midpoint=0, orientation='vertical', rotation=-90,IDcolor = "blue",
                     IDsize =10, **kwargs)
        =============================================================================
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
        Interval : [integer], optional
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
        
        
        fig = plt.figure(60, figsize = Figsize)
        gs = gridspec.GridSpec(nrows = 2, ncols = 2, figure = fig )
        ax = fig.add_subplot(gs[:,:])
        ticks = np.arange(np.nanmin(Arr), np.nanmax(Arr),TicksSpacing)
        
        if ColorScale == 1:
            im = ax.matshow(Arr[:,:,0],cmap=cmap, vmin = np.nanmin(Arr), vmax = np.nanmax(Arr),)
            cbar_kw = dict(ticks = ticks)
        elif ColorScale == 2:
            im = ax.matshow(Arr[:,:,0],cmap=cmap, norm=colors.PowerNorm(gamma=gamma,
                                vmin = np.nanmin(Arr), vmax = np.nanmax(Arr)))
            cbar_kw = dict(ticks = ticks)
        elif ColorScale == 3:
            linthresh=1
            linscale=2
            im = ax.matshow(Arr[:,:,0],cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh,
                                    linscale=linscale, base=np.e,vmin = np.nanmin(Arr), vmax = np.nanmax(Arr)))
            formatter = LogFormatter(10, labelOnlyBase=False) 
            cbar_kw = dict(ticks = ticks, format=formatter)
        elif ColorScale == 4:
            bounds = np.arange(np.nanmin(Arr), np.nanmax(Arr),TicksSpacing)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(Arr[:,:,0],cmap=cmap, norm=norm)
            cbar_kw = dict(ticks = ticks)
        else:
            im = ax.matshow(Arr[:,:,0],cmap=cmap, norm=MidpointNormalize(midpoint=midpoint))
            cbar_kw = dict(ticks = ticks)
            
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=Cbarlength, orientation=orientation,**cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=rotation, va="bottom")
        cbar.ax.tick_params(labelsize=10)
        
        
        day_text = ax.text(Textloc[0],Textloc[1], 'Begining',fontsize= cbarlabelsize)
        ax.set_title(Title,fontsize= titlesize)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_xticks([])
        ax.set_yticks([])
        Indexlist = list()
        
        for x in range(Arr.shape[0]):
                for y in range(Arr.shape[1]):
                    if not np.isnan(Arr[x, y,0]):
                        Indexlist.append([x,y])
                    
        Textlist = list()
        for x in range(NoElem):
            Textlist.append(ax.text(Indexlist[x][1], Indexlist[x][0],
                                  round(Arr[Indexlist[x][0], Indexlist[x][1], 0],2), 
                                  ha="center", va="center", color="w", fontsize=NumSize))
        # Points = list()
        PoitsID = list()
        if 'Points' in kwargs.keys():
            row = kwargs['Points'].loc[:,'cell_row'].tolist()
            col = kwargs['Points'].loc[:,'cell_col'].tolist()
            IDs = kwargs['Points'].loc[:,'id'].tolist()
            Points = ax.scatter(col, row, color=Gaugecolor, s=Gaugesize)
            
            for i in range(len(row)):
                PoitsID.append(ax.text(col[i], row[i], IDs[i], ha="center", 
                                       va="center", color=IDcolor, fontsize=IDsize))
            
        # Normalize the threshold to the images color range.
        if Backgroundcolorthreshold is not None:
            Backgroundcolorthreshold = im.norm(Backgroundcolorthreshold)
        else:
            Backgroundcolorthreshold = im.norm(np.nanmax(Arr))/2.
                    
            
        def init() :
            im.set_data(Arr[:,:,0])
            day_text.set_text('')
            
            output = [im, day_text]
            
            if 'Points' in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs['Points'].loc[:,'cell_row'].tolist()
                col = kwargs['Points'].loc[:,'cell_col'].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)
                
                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])
                
                output = output + PoitsID
                
            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], 0],2)
                    Textlist[x].set_text(val)
                
                output = output + Textlist
            
            return output
        
        
        def animate(i):
            im.set_data(Arr[:,:,i])
            day_text.set_text('Date = '+str(Time[i])[0:10])
            
            output = [im, day_text]
            
            if 'Points' in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs['Points'].loc[:,'cell_row'].tolist()
                col = kwargs['Points'].loc[:,'cell_col'].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)
                
                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])
                
                output = output + PoitsID
                
            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], i],2)
                    kw = dict(color=textcolors[int(im.norm(Arr[Indexlist[x][0], Indexlist[x][1], i]) > Backgroundcolorthreshold)])
                    Textlist[x].update(kw)
                    Textlist[x].set_text(val)
                
                output = output + Textlist
                
            return output
        
        plt.tight_layout()
        # global anim
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames = np.shape(Arr)[2],
                                               interval = Interval, blit = True)
        return anim
    
    @staticmethod
    def rescale(OldValue,OldMin,OldMax,NewMin,NewMax):
        """
        ===================================================================
            rescale(OldValue,OldMin,OldMax,NewMin,NewMax)
        ===================================================================
        this function rescale a value between two boundaries to a new value bewteen two
        other boundaries
        inputs:
            1-OldValue:
                [float] value need to transformed
            2-OldMin:
                [float] min old value
            3-OldMax:
                [float] max old value
            4-NewMin:
                [float] min new value
            5-NewMax:
                [float] max new value
        output:
            1-NewValue:
                [float] transformed new value

        """
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return NewValue

    @staticmethod
    def mycolor(x,min_old,max_old,min_new, max_new):
        """
        ===================================================================
             mycolor(x,min_old,max_old,min_new, max_new)
        ===================================================================
        this function transform the value between two normal values to a logarithmic scale
        between logarithmic value of both boundaries
        inputs:
            1-x:
                [float] new value needed to be transformed to a logarithmic scale
            2-min_old:
                [float] min old value in normal scale
            3-max_old:
                [float] max old value in normal scale
            4-min_new:
                [float] min new value in normal scale
            5-max_new:
                [float] max_new max new value
        output:
            1-Y:
                [int] integer number between new max_new and min_new boundaries
        """

        # get the boundaries of the logarithmic scale
        if min_old == 0.0:
            min_old_log = -7
        else:
            min_old_log = np.log(min_old)

        max_old_log = np.log(max_old)

        if x == 0:
            x_log = -7
        else: 
            x_log = np.log(x)

        y = int(np.round(Visualize.rescale(x_log,min_old_log,max_old_log,min_new,max_new)))

        return y

class MidpointNormalize(colors.Normalize):
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        
        return np.ma.masked_array(np.interp(value, x, y))