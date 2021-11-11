# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:31:11 2020

@author: mofarrag
"""
# from IPython import get_ipython
# get_ipython().magic("reset -f")
import os

# import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import stats as stats
from scipy.stats import genextreme, gumbel_r, norm

# from Hapi.statistics.statisticaltools import StatisticalTools as ST
from Hapi.statistics.distributions import (GEV, ConfidenceInterval, Gumbel,
                                           PlottingPosition)
from Hapi.statistics.statisticaltools import StatisticalTools as st

data = [
    15.999737471905252,
    16.105716234887431,
    17.947809230275304,
    16.147752064149291,
    15.991427126788327,
    16.687542227378565,
    17.125139229445359,
    19.39645340792385,
    16.837044960487795,
    15.804473320190725,
    16.018569387471025,
    16.600876724289019,
    16.161306985203151,
    17.338636901595873,
    18.477371969176406,
    17.897236722220281,
    16.626465201654593,
    16.196548622931672,
    16.013794215070927,
    16.30367884232831,
    17.182106070966608,
    18.984566931768452,
    16.885737663740024,
    16.088051117522948,
    15.790480003140173,
    18.160947973898388,
    18.318158853376037,
]

data2 = [
    144,
    213,
    219,
    242,
    285,
    287,
    295,
    304,
    322,
    337,
    349,
    373,
    374,
    378,
    382,
    383,
    405,
    446,
    452,
    453,
    490,
    505,
    512,
    540,
    540,
    542,
    542,
    549,
    555,
    556,
    565,
    568,
    623,
    634,
    645,
    664,
    683,
    683,
    690,
    703,
    740,
    753,
    800,
    801,
    822,
    856,
    863,
    967,
    981,
    1100,
    1160,
    1190,
    1190,
    1250,
]
#%%
Gdist = Gumbel(data)
# defult parameter estimation method is maximum liklihood method
Param_dist = Gdist.EstimateParameter()
print(Param_dist)
loc = Param_dist[0]
scale = Param_dist[1]
# calculate and plot the pdf
pdf = Gdist.pdf(loc, scale, plot_figure=True)
cdf, _, _ = Gdist.cdf(loc, scale, plot_figure=True)
#%% lmoments
Param_dist = Gdist.EstimateParameter(method="lmoments")
print(Param_dist)
loc = Param_dist[0]
scale = Param_dist[1]
# calculate and plot the pdf
pdf = Gdist.pdf(loc, scale, plot_figure=True)
cdf, _, _ = Gdist.cdf(loc, scale, plot_figure=True)
#%%
# calculate the CDF(Non Exceedance probability) using weibul plotting position
data.sort()
# calculate the F (Non Exceedence probability based on weibul)
cdf_Weibul = PlottingPosition.Weibul(data)
# TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution
Qth = Gdist.TheporeticalEstimate(loc, scale, cdf_Weibul)
# test = stats.chisquare(st.Standardize(Qth), st.Standardize(data),ddof=5)
# calculate the confidence interval
upper, lower = Gdist.ConfidenceInterval(loc, scale, cdf_Weibul, alpha=0.1)
# ProbapilityPlot can estimate the Qth and the lower and upper confidence interval in the process of plotting
fig, ax = Gdist.ProbapilityPlot(loc, scale, cdf_Weibul, alpha=0.1)
#%%
"""
if you want to focus only on high values, you can use a threshold to make the code focus on what is higher
this threshold.
"""
threshold = 17
Param_dist = Gdist.EstimateParameter(
    method="optimization", ObjFunc=Gumbel.ObjectiveFn, threshold=threshold
)
print(Param_dist)
loc = Param_dist[0]
scale = Param_dist[1]
Gdist.ProbapilityPlot(loc, scale, cdf_Weibul, alpha=0.1)
#%%
threshold = 18
Param_dist = Gdist.EstimateParameter(
    method="optimization", ObjFunc=Gumbel.ObjectiveFn, threshold=threshold
)
print(Param_dist)
loc = Param_dist[0]
scale = Param_dist[1]
Gdist.ProbapilityPlot(loc, scale, cdf_Weibul, alpha=0.1)
#%% Generalized Extreme Value (GEV)
Gevdist = GEV(data2)
# default parameter estimation method is maximum liklihood method
Param_dist = Gevdist.EstimateParameter()
print(Param_dist)
shape = Param_dist[0]
loc = Param_dist[1]
scale = Param_dist[2]
# calculate and plot the pdf
pdf = Gevdist.pdf(shape, loc, scale, plot_figure=True)
cdf, _, _ = Gevdist.cdf(shape, loc, scale, plot_figure=True)
#%% lmoment method
Param_dist = Gevdist.EstimateParameter(method="lmoments")
print(Param_dist)
shape = Param_dist[0]
loc = Param_dist[1]
scale = Param_dist[2]
# calculate and plot the pdf
pdf = Gevdist.pdf(shape, loc, scale, plot_figure=True)
cdf, _, _ = Gevdist.cdf(shape, loc, scale, plot_figure=True)
#%%
data.sort()
# calculate the F (Non Exceedence probability based on weibul)
cdf_Weibul = PlottingPosition.Weibul(data)
T = PlottingPosition.Weibul(data, option=2)
# TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution
Qth = Gevdist.TheporeticalEstimate(shape, loc, scale, cdf_Weibul)

func = ConfidenceInterval.GEVfunc
upper, lower = Gevdist.ConfidenceInterval(
    shape, loc, scale, F=cdf_Weibul, alpha=0.1, statfunction=func, n_samples=len(data)
)

#%%
"""
calculate the confidence interval using the boot strap method directly
"""
CI = ConfidenceInterval.BootStrap(
    data, statfunction=func, gevfit=Param_dist, n_samples=len(data), F=cdf_Weibul
)
LB = CI["LB"]
UB = CI["UB"]
#%%
fig, ax = Gevdist.ProbapilityPlot(
    shape, loc, scale, cdf_Weibul, func=func, n_samples=len(data)
)
