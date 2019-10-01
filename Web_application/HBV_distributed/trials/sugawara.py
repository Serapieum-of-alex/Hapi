# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 08:31:08 2015

@author: chaco3

Tank Model

Implemented By Juan Chacon
"""
from __future__ import division
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as opt
#%%
INITIAL_STATES = [10, 10]
INITIAL_Q = 1.0
INITIAL_PARAM = [0.5, 0.2, 0.01, 0.1, 10.0, 20.0, 1, 1]
#INITIAL_PARAM = [0.1819, 0.0412, 0.3348, 0.0448, 3.2259, 0.3800]

PARAM_BND = ((0.0, 1.1),
             (0.0, 1.1),
             (0.0, 1.5),
             (0.0, 1.1),
             (1.0, 15.0),
             (0.1, 1.0),
             (0.8, 1.2),
			 (0.8, 1.2))


def _step(prec, evap, st, param, extra_param):
    '''
    #this function takes the following arguments
    #I -> inputs(2)[prec, evap]
    #    I(1) prec: Precipitation [mm]
    #    I(2) evap: Evaporation [mm]
    #
    #S -> System states(2)[S1, S2]
    #    S(1) S1: Level of the top tank [mm]
    #    S(2) S2: Level of the bottom tank [mm]
    #
    #P -> Parameter vector(6)
    #    P(1) k1: Upper tank upper discharge coefficient
    #    P(2) k2: Upper tank lower discharge coefficient
    #    P(3) k3: Percolation to lower tank coefficient
    #    P(4) k4: Lower tank discharge coefficient
    #    P(5) d1: Upper tank upper discharge position
    #    P(6) d2: Upper tank lower discharge position
    #
    #EP -> Extra parameters(2)
    #    EP(1) DT: Number of hours in the time step [s]
    #    EP(2) AREA: Catchment area [km²]
    #
    #Outputs
    #Q -> Flow [m³/s]
    #S -> Updated system states(2)[S1, S2] mm
    '''

    # Old states
    S1Old = st[0]
    S2Old = st[1]

    #Parameters
    k1 = param[0]
    k2 = param[1]
    k3 = param[2]
    k4 = param[3]
    d1 = param[4]
    d2 = param[5]
    rfcf = param[6]
    ecorr = param[7]
    
    # Extra Parameters
    DT = extra_param[0]
    Area = extra_param[1]

    ## Top tank
    H1 = np.max([S1Old + prec*rfcf - evap*ecorr, 0])

    if H1 > 0:
        #direct runoff
        if H1 > d1:
            q1 = k1*(H1-d1)
        else:
            q1 = 0

        #Fast response component
        if H1 > d2:
            q2 = k2*(H1-d2)
        else:
            q2 = 0

        #Percolation to bottom tank
        q3 = k3 * H1
        #Check for availability of water in upper tank
        q123 = q1+q2+q3
        if q123 > H1:
            q1 = (q1/q123)*H1
            q2 = (q2/q123)*H1
            q3 = (q3/q123)*H1
    else:
        q1 = 0
        q2 = 0
        q3 = 0

    Q1 = q1+q2
    #State update top tank
    S1New = max(H1 - (q1+q2+q3), 0.0)
    
    ## Bottom tank
    H2 = S2Old+q3
    Q2 = k4* H2

    #check if there is enough water
    if Q2 > H2:
        Q2 = H2

    #Bottom tank update
    S2New = H2 - Q2

    ## Total Flow
    # DT = 86400 #number of seconds in a day
    # Area = 2100# Area km²
    if (Q1 + Q2) >= 0:
        Q = (Q1+Q2)*Area/(3.6*DT)
    else:
        Q = 0

    S = [S1New, S2New]
#    if S1New < 0:
#        print('s1 below zero')
    return Q, S

def simulate(prec, evap, param, extra_param):
    '''

    '''
    st = [INITIAL_STATES,]
    q = [10,]

    for i in range(len(prec)):
        step_res = _step(prec[i], evap[i], st[i], param, extra_param)
        q.append(step_res[0])
        st.append(step_res[1])
        
    return q, st

def calibrate(prec, evap, extra_param, q_rec, verbose=False):

    def mod_wrap(param_cal):
        q_sim = simulate(prec[:-1], evap[:-1], param_cal, extra_param)[0]
        try:
            perf_fun = -NSE(q_sim, q_rec)
        except:
            perf_fun = 9999

        if verbose: print -perf_fun
        return perf_fun

    cal_res = opt.minimize(mod_wrap, INITIAL_PARAM, bounds=PARAM_BND,
                           method='L-BFGS-B')

    return cal_res.x, cal_res.fun

def NSE(x,y,q='def',j=2.0):
    """
    Performance Functions
    x - calculated value
    y - recorded value
    q - Quality tag (0-1)
    j - exponent to modify the inflation of the variance (standard NSE j=2)
    """
    x = np.array(x)
    y = np.array(y)
    a = np.sum(np.power(x-y,j))
    b = np.sum(np.power(y-np.average(y),j))
    F = 1.0 - a/b
    return F


'''
TEsting function    
'''
for i in range(1000):
    prec = np.random.uniform(0, 100)
    evap = np.random.uniform(0, 10)
    st = [np.random.uniform(0, 30), np.random.uniform(0, 30)]
    param = [0.1819, 0.0412, 0.3348, 0.0448, 3.2259, 0.3800,1,1, 1]
    extra_param = [1, 145]
    _step(prec, evap, st, param, extra_param)