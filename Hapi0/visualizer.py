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

hours = list(range(1,25))

class Visualize():

    def __init__(self, Sub, resolution = "Hourly"):
        self.resolution = "Hourly"
        # self.XSname = Sub.crosssections['xsid'].tolist()


    def GroundSurface(self, Sub, XS=0, XSbefore = 10, XSafter = 10):

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
        axGS.plot(Sub.XSname, Sub.crosssections['zl'],'k--', dashes = (5,1), linewidth = 2, label = 'Left Dike')
        axGS.plot(Sub.XSname, Sub.crosssections['zr'],'k.-', linewidth = 2, label = 'Right Dike')

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
                            XSbefore = 10, XSafter = 10):
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
            day_text = ax4.annotate('',xy=(Sub.XSname[0],Sub.crosssections['gl'].min()),fontsize= 20)
        else:
            day_text = ax4.annotate('',xy=(FigureFirstXS+1,Sub.crosssections['gl'][FigureLastXS]+1),fontsize= 20)


        WLline, = ax4.plot([],[],linewidth = 5)
        hLline, = ax4.plot([],[],linewidth = 5)


        gs.update(wspace = 0.2, hspace = 0.2, top= 0.96, bottom = 0.1, left = 0.05, right = 0.96)
        # animation

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

            # BC Q (ax2)
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
        plt.tight_layout()

        #Writer = animation.FFMpegWriter
        #Writer= Writer(fps=30, bitrate=1800, #, metadata=dict(artist='Me')
        #               extra_args=['-vcodec', 'libx264'])
        #animation.FFMpegFileWriter(**kwargs = {"outfile" : 'basic_animation.mp4'})

        anim = animation.FuncAnimation(fig, animate_q, init_func=init_q, frames = np.shape(counter)[0],
                                       interval = Interval, blit = True)

        #anim.save('basic_animation.mp4', writer =Writer) #fps=30,
        plt.show()


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
            if Sub.Version == 2:
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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS1.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS1.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS2.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS2.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS3.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS3.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
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
            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS4.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS4.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS5.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS5.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS6.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS6.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS7.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS7.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS8.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS8.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )

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

            if Sub.Version == 2:
                dbf= Sub.crosssections['dbf'].loc[Sub.crosssections['xsid'] == XSS.index[XSno]].values[0]
                ax_XS9.annotate("dbf="+str(round(dbf,2)),xy=(bl,dbf-0.5), fontsize = textsize  )
                ax_XS9.annotate("Area="+str(round(dbf*b,2))+"m2",xy=(bl,dbf-1.4), fontsize = textsize  )
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