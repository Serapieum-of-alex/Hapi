"""
Created on Thu Mar  5 14:41:04 2020

@author: mofarrag
"""
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

import Hapi.hm.calibration as RC
import Hapi.hm.river as R
from Hapi.plot.visualizer import Visualize as V

CompP = r"C:\MyComputer\01Algorithms\Hydrology\Hapi/"
#%% paths
""" these following paths are for the inputs that does not change """
hm_path = CompP + r"examples\Hydrodynamic models\data\hm/"
observed_path = CompP + r"examples\Hydrodynamic models\data\observed/"
gauges_files = CompP + r"examples\Hydrodynamic models\data/Gauges.csv"
rrm_path = CompP + r"examples\Hydrodynamic models\data\rrm/"

start = "1951-1-1"
end = "2003-12-31"
days = 19720
NoValue = -9
Calib = RC.Calibration("HM", 3, gauge_id_col="id")
#%%
Rhine = R.River('HM')
Rhine.statisticalProperties(hm_path + "/Statistical analysis/" + "DistributionProperties.csv",
                            Distibution="Gumbel")

Rhine_obs = R.River('Observed')
Rhine_obs.statisticalProperties(observed_path + "/Statistical analysis/" + "DistributionProperties.csv",
                                Distibution="Gumbel")
#%% read the discharge data to get the max annual values
Calib.readGaugesTable(gauges_files)
# read the gauge flow hydrographs
Calib.readObservedQ(observed_path, start, end, NoValue)#,column="id"
# read the SWIM hydrographs for the gauge sub-basins
Calib.readRRM(rrm_path)
Calib.readHMQ(hm_path)
#%% Get the max annual
Calib.getAnnualMax(option=1)
Calib.getAnnualMax(option=3)
Calib.getAnnualMax(option=4)

cols = Calib.hm_gauges['id'].tolist()
Qgauges = pd.DataFrame()
Qgauges['SubID'] = cols

Calib.RPObs = pd.DataFrame(index = Calib.annual_max_obs_q.index, columns = cols)
Calib.RPHM = pd.DataFrame(index = Calib.annual_max_obs_q.index, columns = cols)

Qgauges = Qgauges.assign(start = 0, end = 0)

# get the start and end date (excluding the gaps) of each gauges
for i in range(len(Qgauges)):
    Sub = Qgauges.loc[i,'SubID']
    st1 = Calib.annual_max_obs_q.loc[:, Sub][Calib.annual_max_obs_q.loc[:, Sub] != NoValue].index[0]
    st2 = Calib.annual_max_hm_q.loc[:, Sub][Calib.annual_max_hm_q.loc[:, Sub] != NoValue].index[0]

    Qgauges.loc[i,'start'] = max(st1,st2)
#    start.append(ObsQ[ObsQ.columns[i]][ObsQ[ObsQ.columns[i]] != NoValue].index[0])
    end1 = Calib.annual_max_obs_q[Sub][Calib.annual_max_obs_q[Sub] != NoValue].index[-1]
    end2 = Calib.annual_max_hm_q[Sub][Calib.annual_max_hm_q[Sub] != NoValue].index[-1]
    Qgauges.loc[i,'end'] =min(end1,end2)

for i in range(len(Qgauges)):
    Sub = Qgauges.loc[i,'SubID']
    fromd = Qgauges.loc[i,'start']
    tod = Qgauges.loc[i,'end']
    # HM
    Qrp = np.array(Calib.annual_max_hm_q.loc[fromd:tod, Sub])
    Calib.RPHM.loc[fromd:tod, Sub] =  Rhine.getReturnPeriod(Sub, Qrp, distribution="Gumbel")
    # Obs
    Qrp = np.array(Calib.annual_max_obs_q.loc[fromd:tod, Sub])
    Calib.RPObs.loc[fromd:tod, Sub] =  Rhine_obs.getReturnPeriod(Sub, Qrp, distribution="Gumbel")

"""
    # to get the Non Exceedance probability for a specific Value
    loc = River2.SP.loc[River2.SP['ID']== Sub,'loc'].tolist()[0]
    scale = River2.SP.loc[River2.SP['ID']== Sub,'scale'].tolist()[0]
    F = gumbel_r.cdf(Qrp, loc=loc, scale=scale)
    T = 1/(1-F)
"""
#%% plot the frequency curves at the location of the gaues for HM and historical data
# for the 'RP2','RP5','RP10','RP15','RP20','RP50','RP100' only
# SubID = 53
color1 = '#27408B'
color2 = '#DC143C'
color3 = "grey"

for i in range(len(Qgauges)):
    SubID = Qgauges.loc[i, 'SubID']
    fig, ax1 = plt.subplots(figsize=(6.3,4.2))
    xlabels = ['RP2','RP5','RP10','RP15','RP20','RP50','RP100']

    ax1.plot(xlabels, Rhine_obs.SP.loc[Rhine_obs.SP['id'] == SubID, xlabels].values.tolist()[0],
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = 5, label = "GRDC")

    ax1.plot(xlabels, Rhine.SP.loc[Rhine.SP['id'] == SubID, xlabels].values.tolist()[0],
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = 5, label = "HM ")

    ax1.set_ylabel("Discharge (m3/s)", fontsize = 15)
    ax1.set_xlabel("Return Period", fontsize = 15)
    fig.legend(loc="upper right", bbox_to_anchor=(0.3,1), bbox_transform=ax1.transAxes,fontsize = 12)
    #plt.title("Inundated Area ( 1000 cell)", fontsize = 15)
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()
    """
    uncomment the saving line
    """
    # plt.savefig(SaveTo + "/" + str(SubID) + ".tif",transparent=True, framealpha=0)
    plt.close()
#%% calculate the return periods for the whole 1000 year for all HM, historical data
# return periods
T = np.linspace(2,1000,333)
# T = np.array([2,5,10,15,20,50,100])
F = 1-(1/T)
# T = [int(i) for i in T]

Rhine_obs.Qrp = pd.DataFrame(columns=T, index = Rhine_obs.SP['id'].tolist())
# River1.Qrp = pd.DataFrame(columns=T, index = GRDC.SP['id'].tolist())
Rhine.Qrp = pd.DataFrame(columns=T, index = Rhine_obs.SP['id'].tolist())

for i in range(len(Qgauges)):
    SubID = Qgauges.loc[i, 'SubID']
    # GRDC.Qrp.loc[GRDC.Qrp.index[i],:] = gumbel_r.ppf(F,loc=GRDC.SP.loc[i,"loc"], scale=GRDC.SP.loc[i,"scale"])
    # River1.Qrp.loc[River1.Qrp.index[i],:] = gumbel_r.ppf(F,loc=River1.SP.loc[i,"loc"], scale=River1.SP.loc[i,"scale"])
    # River2.Qrp.loc[River2.Qrp.index[i],:] = gumbel_r.ppf(F,loc=River2.SP.loc[i,"loc"], scale=River2.SP.loc[i,"scale"])
    Rhine_obs.Qrp.loc[SubID, :] = Rhine_obs.getQForReturnPeriod(SubID, T, distribution="Gumbel")
    # River1.Qrp.loc[River1.Qrp.index[i],:] = River1.GetQForReturnPeriod(GRDC.SP.loc[i,"id"], T)
    Rhine.Qrp.loc[SubID, :] = Rhine.getQForReturnPeriod(SubID, T, distribution="Gumbel")
#%% plot the frequency curves separately
color1 = '#27408B'
color2 = '#DC143C'
color3 = "grey"

for i in range(len(Qgauges)):
    SubID = Qgauges.loc[i, 'SubID']
    # SubID = Rhine_obs.SP.loc[i, 'id']
    fig, ax1 = plt.subplots(figsize=(8.6,4.7))
    # xlabels = ['RP2','RP5','RP10','RP15','RP20','RP50','RP100']

    ax1.plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = 5, label = "Observed")

    # ax1.plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
    #           zorder=1, color = color1 , linestyle = V.LineStyle(6),
    #           linewidth = 5, label = "RIM1.0")


    ax1.plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = 5, label = "HM")


    ax1.set_ylabel("Discharge (m3/s)", fontsize = 15)
    ax1.set_xlabel("Return Period", fontsize = 15)
    # fig.legend(loc="upper right", bbox_to_anchor=(0.3,1), bbox_transform=ax1.transAxes,fontsize = 12)
    #plt.title("Inundated Area ( 1000 cell)", fontsize = 15)
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()
    """
    uncomment the saving line
    """
    # plt.savefig(SaveTo + "/new-" + str(SubID) + ".tif",transparent=True, framealpha=0)
    # plt.close()
#%% frequency curves in subplot
color1 = '#27408B'
color2 = '#DC143C'
color3 = "grey"

titlesize = 15
labelsize = 12
linewidth = 4
markersize = 7
PlotPoint = True

"""
if you get an error
rerun the block "Get the max annual"
"""

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12,8))

SubID=1

ax[0,0].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=3, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[0,0].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[0,0].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=2, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[0,0].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[0,0].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[0,0].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[0,0].set_xscale('log')
ax[0,0].set_title("Rees", fontsize=titlesize)
ax[0,0].set_yticks([5000,10000,15000,20000])
# ax[0,0].set_xticks([2,200,400,600,800,1000])
ax[0,0].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[0,0].set_xlabel("Return Period (year)", fontsize = labelsize)

SubID=42
ax[0,1].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[0,1].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[0,1].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

ax[0,1].set_yticks([6000,10000,15000,20000])
# ax[0,1].set_xticks([2,200,400,600,800,1000])
# ax[0,1].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[0,1].set_xlabel("Return Period (year)", fontsize = labelsize)

# start, end = ax[0,1].get_ylim()
# ax[0,1].yaxis.set_ticks(np.linspace(start,end,4))
# ax[0,1].set_yticks([5000,10000,15000,20000]) # ax[0,1].get_yticks()
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[0,1].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[0,1].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[0,1].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[0,1].set_xscale('log')
ax[0,1].set_title("Duesseldorf", fontsize=titlesize)

SubID=81
ax[0,2].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[0,2].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[0,2].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

ax[0,2].set_yticks([6000,10000,15000,20000])
# ax[0,2].set_xticks([2,200,400,600,800,1000])
# ax[0,2].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[0,2].set_xlabel("Return Period (year)", fontsize = labelsize)
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[0,2].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[0,2].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[0,2].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[0,2].set_xscale('log')
ax[0,2].set_title("Cologne", fontsize=titlesize)



SubID=132
ax[1,0].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[1,0].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[1,0].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

ax[1,0].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[1,0].set_xlabel("Return Period (year)", fontsize = labelsize)

ax[1,0].set_yticks([4000,8000,12000,16000])
# ax[1,0].set_xticks([2,200,400,600,800,1000])
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[1,0].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[1,0].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[1,0].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[1,0].set_xscale('log')
ax[1,0].set_title("Andernach", fontsize=titlesize)

SubID=461
ax[1,1].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[1,1].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[1,1].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

# ax[1,1].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[1,1].set_xlabel("Return Period (year)", fontsize = labelsize)

ax[1,1].set_yticks([3000,5500,7500,10000])
# ax[1,1].set_xticks([2,200,400,600,800,1000])
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[1,1].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[1,1].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[1,1].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[1,1].set_xscale('log')
ax[1,1].set_title("Speyer", fontsize=titlesize)

SubID=333
ax[1,2].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[1,2].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[1,2].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

# ax[1,2].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
# ax[1,2].set_xlabel("Return Period (year)", fontsize = labelsize)

ax[1,2].set_yticks([1500,3500,5500,8000])
# ax[1,2].set_xticks([2,200,400,600,800,1000])
#
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[1,2].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[1,2].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[1,2].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[1,2].set_xscale('log')
ax[1,2].set_title("Dhrontalsperre", fontsize=titlesize)

SubID=250
ax[2,0].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[2,0].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[2,0].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

ax[2,0].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
ax[2,0].set_xlabel("Return Period (year)", fontsize = labelsize)

ax[2,0].set_yticks([750,1500,2250,3000])
# ax[2,0].set_xticks([2,200,400,600,800,1000])
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[2,0].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[2,0].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[2,0].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[2,0].set_xscale('log')
ax[2,0].set_title("Steinbach", fontsize=titlesize)

SubID=543
ax[2,1].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[2,1].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[2,1].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")
ax[2,1].set_yticks([750,1500,2250,3000])
# ax[2,1].set_xticks([2,200,400,600,800,1000])
# ax[2,1].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
ax[2,1].set_xlabel("Return Period (year)", fontsize = labelsize)

# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[2,1].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[2,1].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[2,1].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[2,1].set_xscale('log')
ax[2,1].set_title("Lauffen", fontsize=titlesize)

SubID=136
ax[2,2].plot(Rhine_obs.Qrp.columns.tolist(), Rhine_obs.Qrp.loc[SubID, :].values.tolist(),
             zorder=5, color = color3, linestyle = V.LineStyle(0),
             linewidth = linewidth, label = "Observed")

# ax[2,2].plot(River1.Qrp.columns.tolist(),River1.Qrp.loc[SubID,:].values.tolist(),
#               zorder=1, color = color1 , linestyle = V.LineStyle(6),
#               linewidth = linewidth, label = "RIM1.0")


ax[2,2].plot(Rhine.Qrp.columns.tolist(), Rhine.Qrp.loc[SubID, :].values.tolist(),
             zorder=1, color = color2, linestyle = V.LineStyle(9),
             linewidth = linewidth, label = "HM")

ax[2,2].set_yticks([200,450,700,950])
# ax[2,2].set_xticks([2,200,400,600,800,1000])
# ax[2,2].set_ylabel("Discharge (m3/s)", fontsize = labelsize)
ax[2,2].set_xlabel("Return Period (year)", fontsize = labelsize)
# points
if PlotPoint :
    fromd = Qgauges.loc[Qgauges['SubID'] == SubID,'start'].tolist()[0]
    tod = Qgauges.loc[Qgauges['SubID'] == SubID,'end'].tolist()[0]

    ax[2,2].plot(Calib.RPObs.loc[fromd:tod, SubID].tolist(), Calib.annual_max_obs_q.loc[fromd:tod, SubID].tolist(),
                 'o', zorder=6, color = 'black', markersize = markersize, fillstyle='none') #markerfacecoloralt=""
    ax[2,2].plot(Calib.RPHM.loc[fromd:tod, SubID].tolist(), Calib.annual_max_hm_q.loc[fromd:tod, SubID].tolist(),
                 'x', zorder=4, color = 'black', markersize = markersize, fillstyle='none')
    # ax[2,2].plot(RIM1.RPRIM.loc[fromd:tod,SubID].tolist(), RIM1.AnnualMaxRIMQ.loc[fromd:tod,SubID].tolist(),
    #              's',zorder=5, color = 'black', markersize = markersize, fillstyle='none')

ax[2,2].set_xscale('log')
ax[2,2].set_title("Leun", fontsize=titlesize)

plt.rcParams.update({'font.size': 11})
ax[0,2].legend(bbox_to_anchor=(0.03, 0.95), loc='upper left', borderaxespad=0.)
# fig.tight_layout() #,pad=1.5
plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.95, bottom=0.07, left=0.07, right=0.95)
# plt.savefig(SaveTo+"/trend-log-points.tif",transparent=True, framealpha=0)
