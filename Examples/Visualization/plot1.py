# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:16:53 2021

@author: mofarrag
"""
import matplotlib.pyplot as plt
import numpy as np
import Hapi.visualizer as V
Vis = V.Visualize(1)
from Hapi.statisticaltools  import StatisticalTools as ST
#%%
distance = [19.5, 43.0, 71.0, 123.5, 164.5, 204.5, 212.5, 222.0, 251.0, 343.0, 483.5, 679.0, 921.5]

wl1 = [136.82, 132.7, 123.05, 115.27, 101.56, 81.12, 68.76, 66.36, 65.84, 50.06, 39.65, 29.59, 22.66]

wl2 = [134.64, 134.11, 120.87, 115.89, 94.8, 79.13, 67.85, 65.63, 64.61, 46.64, 37.93, 27.99, 22.02]

diff = np.random.uniform(-0.5, 0.5, size=len(wl1))

OT1 = np.random.uniform(5000, 20000, size=len(wl1))
OT2 = np.random.uniform(7000, 16000, size=len(wl1))
#%%

def Plot_Type1(X_axis, Y_axis1, Y2_axis2, Points, PointsY, PointMaxSize=200,
               PointMinSize=1, X_axis_label='X Axis', LegendNum=5, LegendLoc = (1.3, 1),
               PointLegendTitle="Output 2", Ylim=[0,180], Y2lim=[-2,14],
               color1 = '#27408B', color2 = '#DC143C', color3 = "grey",
               linewidth = 4, **kwargs): #Y2_axis2
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10,6))


    ax2 = ax1.twinx()

    ax1.plot(X_axis , Y_axis1, zorder=1, color =color1 ,
              linestyle = Vis.LineStyle(0), linewidth = linewidth,
              label = "Model 1 Output1")

    if 'Y_axis1_2' in kwargs.keys():
        Y_axis1_2 = kwargs['Y_axis1_2']

        rows_axis1, cols_axis1 = np.shape(Y_axis1_2)

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = ['label'] * (cols_axis1 - 1)

        for i in range(1, cols_axis1):
            ax1.plot(Y_axis1_2[:,0] , Y_axis1_2[:,i], zorder=1, color = color2 ,
                      linestyle = Vis.LineStyle(3), linewidth = linewidth,
                      label = label[i-1])

    ax2.plot(X_axis , Y2_axis2, zorder=1, color =color3 ,
              linestyle = Vis.LineStyle(6), linewidth = 2,
              label = "Output1-Diff")

    if 'Y_axis2_2' in kwargs.keys():
        Y_axis2_2 = kwargs['Y_axis2_2']
        rows_axis2, cols_axis2 = np.shape(Y_axis2_2)

        label = kwargs['Y_axis2_label']

        for i in range(1, rows_axis2):
            ax1.plot(Y_axis2_2[0,:] , Y_axis2_2[i,:], zorder=1, color = color2 ,
                      linestyle = Vis.LineStyle(3), linewidth = linewidth,
                      label = label[i-1])

    if 'Points1' in kwargs.keys():
        Points1 = kwargs['Points1']
        vmax = np.max(Points1)
        vmin = np.min(Points1)

        vmax = max(Points.max(), vmax)
        vmin = min(Points.min(), vmin)

    else:
        vmax = max(Points)
        vmin = min(Points)

    vmaxnew = PointMaxSize
    vminnew = PointMinSize

    Points_scaled = [ST.Rescale(x,vmin,vmax,vminnew, vmaxnew)   for x in Points]
    if 'Points1' in kwargs.keys():
        row_points, col_points = np.shape(Points1)
        for i in range(col_points):

            Points1_scaled = [ST.Rescale(x,vmin,vmax,vminnew, vmaxnew)   for x in Points1[:,i]]


    f1 = np.ones(shape=(len(Points)))*PointsY

    if 'Points1' in kwargs.keys():
        PointsY1 = kwargs['PointsY1']
        f2 = np.ones_like(Points1)
        for i in range(len(Points1)):
            f2[:,i] = PointsY1[i]

    scatter = ax2.scatter(X_axis, f1, zorder=1, c=color1 ,
                s = Points_scaled, label = "Model 1 Output 2")

    if 'Points1' in kwargs.keys():
        ax2.scatter(X_axis, f2, zorder=1, c = color2 ,
                   s = Points1_scaled, label = "Model 2 Output 2")

    # produce a legend with the unique colors from the scatter
    legend1 = ax2.legend(*scatter.legend_elements(),
                         bbox_to_anchor=(1.1, 0.2)) #loc="lower right", title="RIM"

    ax2.add_artist(legend1)

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6,num=LegendNum)
    # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
    L = [float(i[14:-2]) for i in labels]
    labels1 = [round(ST.Rescale(x,vminnew, vmaxnew,vmin,vmax)/1000)   for x in L]

    legend2 = ax2.legend(handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle)
    ax2.add_artist(legend2)

    ax1.set_ylim(Ylim)
    ax2.set_ylim(Y2lim)
    #
    ax1.set_ylabel('Output 1 (m)', fontsize = 12)
    ax2.set_ylabel('Output 1 - Diff (m)', fontsize = 12)
    ax1.set_xlabel(X_axis_label, fontsize = 12)
    ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
    ax1.tick_params(which='minor', length=5)
    fig.legend(loc="lower center", bbox_to_anchor=(1.3,0.3), bbox_transform=ax1.transAxes,fontsize = 10)
    plt.rcParams.update({'ytick.major.size': 3.5})
    plt.rcParams.update({'font.size': 12})
    plt.title("Model Output Comparison", fontsize = 15)

    plt.subplots_adjust(right=0.7)
    # plt.tight_layout()

    return (ax1,ax2), fig

#%%
Y_axis1_2 = np.transpose([distance, wl2])
# Y_axis1_2 = np.transpose(Y_axis1_2)

# label = ['YYY']
Points1 = np.transpose([OT2])
PointsY1 = [13]

Plot_Type1(distance, wl1, diff, OT1, 13, PointMaxSize=200, PointMinSize=1,
               X_axis_label='Distance', LegendNum=5, LegendLoc = (1.3, 1),
               PointLegendTitle="Output 2", Ylim=[0,180], Y2lim=[-2,14],
               Y_axis1_2=Y_axis1_2, Points1 = Points1, PointsY1 = PointsY1)#label= label,