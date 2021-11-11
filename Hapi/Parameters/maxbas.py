# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 02:04:13 2018

@author: Mostafa
"""
#% links
from IPython import get_ipython  # to reset the variable explorer each time

get_ipython().magic("reset -f")
import os

import HBV_explicit
import matplotlib.pyplot as plt
#%% library
import numpy as np
# import plotly.plotly as py
import plotly.tools as tls
from matplotlib import gridspec

import Hapi.Routing as RT

# os.chdir("F:/02Private/02Research/thesis/My Thesis/Data_and_Models/Model/Code/05Distributed/")


# import sys
# sys.path.append("F:/02Private/02Research/thesis/My Thesis/Data_and_Models/Model/Code/python_functions")


#%%
Q = np.array([1.5, 1, 2, 6, 10, 20, 15, 7, 3, 2, 1.5, 2])
n1 = RT.Tf(1)

# HBV_explicit._tf(1.5)
# n2=HBV_explicit._tf(2)
n3 = RT.Tf(3)
# n4=HBV_explicit._tf(4)
n5 = RT.Tf(5)
# n6=HBV_explicit._tf(6)
n7 = RT.Tf(7)
n9 = RT.Tf(9)
n12 = RT.Tf(12)

Q3 = RT.TriangularRouting(Q, 3)
Q5 = RT.TriangularRouting(Q, 5)
Q7 = RT.TriangularRouting(Q, 7)
Q9 = RT.TriangularRouting(Q, 9)
Q12 = RT.TriangularRouting(Q, 12)

#%% plot
# plt.figure(1,figsize=(15,8))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
# plt.plot(n1, label="MAXBAS=1",linewidth=3)
# plt.plot(n2, label="MAXBAS=2",linewidth=3)
ax1.bar(range(len(n3)), n3, label="MAXBAS=3", linewidth=3, color="#DC143C")
ax1.legend(fontsize=15)

ax2.bar(range(len(n5)), n5, label="MAXBAS=5", linewidth=3, color="#DC143C")
ax2.legend(fontsize=15)

ax3.bar(range(len(n7)), n7, label="MAXBAS=7", linewidth=3, color="#DC143C")
ax3.legend(fontsize=15)

ax4.bar(range(len(n9)), n9, label="MAXBAS=9", linewidth=3, color="#DC143C")
ax4.legend(fontsize=15)

# plt.xlabel("Time step (hour)")
# plt.ylabel("Discharge m3/s")
#%%
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
# plt.plot(n1, label="MAXBAS=1",linewidth=3)
# plt.plot(n2, label="MAXBAS=2",linewidth=3)
ax1.bar(range(len(n12)), n12, linewidth=3, color="#DC143C")  # label="MAXBAS=3"
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.legend(fontsize=15)
# fig.savefig("maxbas.tif", transparent=True)
#%%
fig = plt.figure(5, figsize=(15, 8))
gs = gridspec.GridSpec(2, 5)

ax1 = fig.add_subplot(gs[0, 3])
ax1.bar(range(len(n3)), n3, label="MAXBAS=3", linewidth=3, color="#3D59AB")
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[0, 4])
ax2.bar(range(len(n5)), n5, label="MAXBAS=5", linewidth=3, color="#DC143C")
ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[1, 3])
ax3.bar(range(len(n7)), n7, label="MAXBAS=7", linewidth=3, color="#66CD00")
ax3.legend(fontsize=9)

ax4 = fig.add_subplot(gs[1, 4])
ax4.bar(range(len(n9)), n9, label="MAXBAS=9", linewidth=3, color="#FFC125")
ax4.legend(fontsize=9)

ax5 = fig.add_subplot(gs[0:, 0:3])
ax5.plot(Q, label="Discharge", linewidth=3, color="#FF34B3")  # , color="#DC143C"
ax5.plot(Q3, label="MAXBAS=3", linewidth=3, color="#3D59AB")
ax5.plot(Q5, label="MAXBAS=5", linewidth=3, color="#DC143C")
ax5.plot(Q7, label="MAXBAS=7", linewidth=3, color="#66CD00")
ax5.plot(Q9, label="MAXBAS=9", linewidth=3, color="#FFC125")
ax5.plot(Q12, label="MAXBAS=12", linewidth=3, color="#8E8E8E")
ax5.set_xlabel("Time step", fontsize=15)
ax5.set_ylabel("Discharge m3/s", fontsize=15)
ax5.legend(fontsize=15)
gs.update(wspace=0.35, hspace=0.1)
#%%
fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig["layout"]["title"] = "Subplots with variable widths and heights"
plotly_fig["layout"]["margin"].update({"t": 40})
plot_url = py.plot(plotly_fig, filename="mpl-subplot-variable-width")
#%%
plt.xlabel("Time step (hour)")
plt.ylabel("Discharge m3/s")
plt.legend(fontsize=15)
