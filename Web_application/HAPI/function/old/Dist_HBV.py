"""
======
Conceptual Distibuted HBV model
======

Distibuted hydrological model.

This is the HBV-96 implementation by Juan Chacon at IHE-Delft, NL. This code
implements the HBV-96 version, as described in Lindstrom et al (1997)
https://doi.org/10.1016/S0022-1694(97)00041-3

"""
# libraries
from __future__ import division, print_function
import numpy as np
import scipy.optimize as opt
import gdal

# functions
import Routing

#import pygeoprocessing.routing as rt
#import pygeoprocessing.geoprocessing as gp

# HBV base model parameters
P_LB = [-1.5, #ltt
        0.001, #utt
        0.001, #ttm
        0.04, #cfmax [mm c^-1 h^-1]
        50.0, #fc
        0.6, #ecorr
        0.001, #etf
        0.2, #lp
        0.00042, #k [h^-1] upper zone
        0.0000042, #k1 lower zone
        0.001, #alpha
        1.0, #beta
        0.001, #cwh
        0.01, #cfr
        0.0, #c_flux
        0.001, #perc mm/h
        0.6, #rfcf
        0.4,  #sfcf
        1] # Maxbas

P_UB = [2.5, #ttm
        3.0, #utt
        2.0, #ttm
        0.4, #cfmax [mm c^-1 h^-1]
        500.0, #fc
        1.4, #ecorr
        5.0, #etf
        0.5, #lp
        0.0167, #k upper zone
        0.00062, #k1 lower zone
        1.0, #alpha
        6.0, #beta
        0.1, #cwh
        1.0, #cfr
        0.08, #c_flux - 2mm/day
        0.125, #perc mm/hr
        1.4, #rfcf
        1.4, #sfcf
        10] # maxbas

DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0]
DEF_q0 = 2.3

# Get random parameter set
def get_random_pars():
    return np.random.uniform(P_LB, P_UB)
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac, pcorr):
    '''
    ==============
    Precipitation
    ==============

    Precipitaiton routine of the HBV96 model.

    If temperature is lower than ltt, all the precipitation is considered as
    snow. If the temperature is higher than utt, all the precipitation is
    considered as rainfall. In case that the temperature is between ltt and
    utt, precipitation is a linear mix of rainfall and snowfall.

    Parameters
    ----------
    temp : float
        Measured temperature [C]
    ltt : float
        Lower temperature treshold [C]
    utt : float
        Upper temperature treshold [C]
    prec : float 
        Precipitation [mm]
    rfcf : float
        Rainfall corrector factor
    sfcf : float
        Snowfall corrector factor

    Returns
    -------
    _rf : float
        Rainfall [mm]
    _sf : float
        Snowfall [mm]
    '''

    if temp <= ltt:  # if temp <= lower temp threshold 
        _rf = 0.0         #no rainfall all the precipitation will convert into snowfall
        _sf = prec*sfcf

    elif temp >= utt: # if temp > upper threshold 
        _rf = prec*rfcf # no snowfall all the precipitation becomes rainfall 
        _sf = 0.0

    else:               # if  ltt< temp < utt
        _rf = ((temp-ltt)/(utt-ltt)) * prec * rfcf
        _sf = (1.0-((temp-ltt)/(utt-ltt))) * prec * sfcf
        
    _rf=_rf*pcorr

    return _rf, _sf

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _snow(cfmax, tfac, temp, ttm, cfr, cwh, _rf, _sf, wc_old, sp_old):
    '''
    ====
    Snow
    ====
    
    Snow routine of the HBV-96 model.
    
    The snow pack consists of two states: Water Content (wc) and Snow Pack 
    (sp). The first corresponds to the liquid part of the water in the snow,
    while the latter corresponds to the solid part. If the temperature is 
    higher than the melting point, the snow pack will melt and the solid snow
    will become liquid. In the opposite case, the liquid part of the snow will
    refreeze, and turn into solid. The water that cannot be stored by the solid
    part of the snow pack will drain into the soil as part of infiltration.

    Parameters
    ----------
    cfmax : float 
        Day degree factor
    tfac : float
        Temperature correction factor
    temp : float 
        Temperature [C]
    ttm : float 
        Temperature treshold for Melting [C]
    cfr : float 
        Refreezing factor
    cwh : float 
        Capacity for water holding in snow pack
    _rf : float 
        Rainfall [mm]
    _sf : float 
        Snowfall [mm]
    wc_old : float 
        Water content in previous state [mm]
    sp_old : float 
        snow pack in previous state [mm]

    Returns
    -------
    _in : float 
        Infiltration [mm]
    _wc_new : float 
        Water content in posterior state [mm]
    _sp_new : float 
        Snowpack in posterior state [mm]
    '''

    if temp > ttm:# if temp > melting threshold
        # then either some snow will melt or the entire snow will melt 
        if cfmax*(temp-ttm) < sp_old+_sf: #if amount of melted snow < the entire existing snow (previous amount+new)
            _melt = cfmax*(temp-ttm)
        else:                             #if amount of melted snow > the entire existing snow (previous amount+new)
            _melt = sp_old+_sf           # then the entire existing snow will melt (old snow pack + the current snowfall)

        _sp_new = sp_old + _sf - _melt
        _wc_int = wc_old + _melt + _rf

    else:                               # if temp < melting threshold
        #then either some water will freeze or all the water willfreeze
        if cfr*cfmax*(ttm-temp) < wc_old+_rf: #then either some water will freeze or all the water willfreeze
            _refr = cfr*cfmax*(ttm - temp)  #cfmax*(ttm-temp) is the rate of melting of snow while cfr*cfmax*(ttm-temp) is the rate of freeze of melted water  (rate of freezing > rate of melting) 
        else:                               # if the amount of frozen water > entire water available
            _refr = wc_old + _rf

        _sp_new = sp_old + _sf + _refr
        _wc_int = wc_old - _refr + _rf

    if _wc_int > cwh*_sp_new: # if water content > holding water capacity of the snow
        _in = _wc_int-cwh*_sp_new  #water content  will infiltrate
        _wc_new = cwh*_sp_new # and the capacity of snow of holding water will retained
    else:           # if water content < holding water capacity of the snow
        _in = 0.0            # no infiltration
        _wc_new = _wc_int

    return _in, _wc_new, _sp_new
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""

def _soil(fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf,
          ep, sm_old, uz_old):
    '''
    ====
    Soil
    ====
    
    Soil routine of the HBV-96 model.
    
    The model checks for the amount of water that can infiltrate the soil, 
    coming from the liquid precipitation and the snow pack melting. A part of 
    the water will be stored as soil moisture, while other will become runoff, 
    and routed to the upper zone tank.

    Parameters
    ----------
    fc : float 
        Filed capacity
    beta : float 
        Shape coefficient for effective precipitation separation
    etf : float 
        Total potential evapotranspiration
    temp : float 
        Temperature
    tm : float 
        Average long term temperature
    e_corr : float 
        Evapotranspiration corrector factor
    lp : float _soil 
        wilting point
    tfac : float 
        Time conversion factor
    c_flux : float 
        Capilar flux in the root zone
    _in : float 
        actual infiltration
    ep : float 
        actual evapotranspiration
    sm_old : float 
        Previous soil moisture value
    uz_old : float 
        Previous Upper zone value

    Returns
    -------
    sm_new : float 
        New value of soil moisture
    uz_int_1 : float 
        New value of direct runoff into upper zone
    '''

    qdr = max(sm_old + inf - fc, 0)  # direct run off as soil moisture exceeded the field capacity
#    qdr=0
    _in = inf - qdr
    _r = ((sm_old/fc)** beta) * _in   # recharge to the upper zone
    
    _ep_int = (1.0 + etf*(temp - tm))*e_corr*ep  # Adjusted potential evapotranspiration
    
    _ea = min(_ep_int, (sm_old/(lp*fc))*_ep_int)

    _cf = c_flux*((fc - sm_old)/fc) # capilary rise
    
    # if capilary rise is more than what is available take all the available and leave it empty
    
    if uz_old + _r < _cf: 
        _cf= uz_old + _r
        uz_int_1=0
    else:
#        uz_int_1 = uz_old + _r - _cf
        uz_int_1 = uz_old + _r - _cf + qdr
    
    sm_new = max(sm_old + _in - _r + _cf - _ea, 0)
    
#    uz_int_1 = uz_old + _r - _cf + qdr

    return sm_new, uz_int_1
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _response(tfac, perc, alpha, k, k1, area, lz_old, uz_int_1):
    '''
    ========
    Response
    ========
    The response routine of the HBV-96 model.
    
    The response routine is in charge of transforming the current values of 
    upper and lower zone into discharge. This routine also controls the 
    recharge of the lower zone tank (baseflow). The transformation of units 
    also occurs in this point.
    
    Parameters
    ----------
    tfac : float
        Number of hours in the time step
    perc : float
        Percolation value [mm\hr]
    alpha : float
        Response box parameter
    k : float
        Upper zone recession coefficient
    k1 : float 
        Lower zone recession coefficient
    area : float
        Catchment area [Km2]
    lz_old : float 
        Previous lower zone value [mm]
    uz_int_1 : float 
        Previous upper zone value before percolation [mm]
    qdr : float
        Direct runoff [mm]
    
    '''    
    # upper zone 
    # if perc > Quz then perc = Quz and Quz = 0 if not perc = value and Quz= Quz-perc so take the min 
    uz_int_2 = np.max([uz_int_1 - perc, 0.0])
    _q_0 = k*(uz_int_2**(1.0 + alpha))
    
    if _q_0 > uz_int_2: # if q_0 =30 and UZ=20
        _q_0= uz_int_2  # q_0=20 
        
    uz_new = uz_int_2 - (_q_0) 
    
    lz_int_1 = lz_old + np.min([perc, uz_int_1])  # if the percolation > upper zone Q all the Quz will percolate
    
    _q_1 = k1*lz_int_1
    
    if _q_1 > lz_int_1:
        _q_1=lz_int_1
    
    lz_new = lz_int_1 - (_q_1)

    q_new = area*(_q_0 + _q_1)/(3.6*tfac)  # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
                                                    # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25

    return q_new, uz_new, lz_new, uz_int_2, lz_int_1

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _tf(maxbas):
    ''' Transfer function weight generator '''
    wi = []
    for x in range(1, maxbas+1):
        if x <= (maxbas)/2.0:
            # Growing transfer
            wi.append((x)/(maxbas+2.0))
        else:
            # Receding transfer
            wi.append(1.0 - (x+1)/(maxbas+2.0))
    
    #Normalise weights
    wi = np.array(wi)/np.sum(wi)
    return wi
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _routing(q, maxbas=1):
    """
    This function implements the transfer function using a triangular 
    function
    """
    assert maxbas >= 1, 'Maxbas value has to be larger than 1'
    # Get integer part of maxbas
#    maxbas = int(maxbas)
    maxbas = int(round(maxbas,0))
    
    # get the weights
    w = _tf(maxbas)
    
    # rout the discharge signal
    q_r = np.zeros_like(q, dtype='float64')
    q_temp = q
    for w_i in w:
        q_r += q_temp*w_i
        q_temp = np.insert(q_temp, 0, 0.0)[:-1]

    return q_r

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _step_run(p, p2, v, St, extra_out=False):
    '''
    ========
    Step run
    ========
    
    Makes the calculation of next step of discharge and states
    
    Parameters
    ----------
    p : array_like [18]
        Parameter vector, set up as:
        [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1, 
        alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
    p2 : array_like [2]
        Problem parameter vector setup as:
        [tfac, area]
    v : array_like [4]
        Input vector setup as:
        [prec, temp, evap, llt]
    St : array_like [5]
        Previous model states setup as:
        [sp, sm, uz, lz, wc]

    Returns
    -------
    q_new : float
        Discharge [m3/s]
    St : array_like [5]
        Posterior model states
    '''
    ## Parse of parameters from input vector to model
    #picipitation function
    
    ltt = 1.0   #p[0] # less than utt and less than lowest temp to prevent sf formation
    utt = 2.0  #p[1]  #very low but it does not matter as temp is 25 so it is greater than 2
    rfcf = 1.0 #p[16] # all precipitation becomes rainfall
    sfcf = 0.00001  #p[17] # there is no snow
    # snow function
    ttm = 1    #p[2] #should be very low lower than lowest temp as temp is 25 all the time so it does not matter
    cfmax = 0.00001  #p[3] as there is no melting  and sp+sf=zero all the time so it doesn't matter the value of cfmax
    cwh = 0.00001     #p[12] as sp is always zero it doesn't matter all wc will go as inf 
    cfr = 0.000001     #p[13] as temp > ttm all the time so it doesn't matter the value of cfr but put it zero
    #soil function
    fc = p[0]       
    beta = p[1]     
    e_corr =1 #p[2]
    etf = p[2]       
    lp =p[3]        
    c_flux =p[4]   
    # response function
    k = p[5]
    k1 = p[6]
    alpha = p[7]
    perc = p[8]
    pcorr=p[9]
    
    ## Non optimisable parameters
    tfac = p2[0] # tfac=0.25
    area = p2[1] #jiboa_area=p2[1] # AREA = 432

    ## Parse of Inputs
    avg_prec = v[0] # Precipitation [mm]
    temp = v[1] # Temperature [C]
    ep = v[2] # Long terms (monthly) Evapotranspiration [mm]
    tm = v[3] #Long term (monthly) average temperature [C]

    ## Parse of states
    sp_old = St[0]
    sm_old = St[1]
    uz_old = St[2]
    lz_old = St[3]
    wc_old = St[4]

    rf, sf = _precipitation(temp, ltt, utt, avg_prec, rfcf, sfcf, tfac, pcorr)
    inf, wc_new, sp_new = _snow(cfmax, tfac, temp, ttm, cfr, cwh, rf, sf,
                               wc_old, sp_old)
    sm_new, uz_int_1 = _soil(fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux,
                            inf, ep, sm_old, uz_old)
    q_new, uz_new, lz_new, uz_int_2, lz_int_1 = _response(tfac, perc, alpha, 
                                                          k, k1, area, lz_old,
                                                          uz_int_1)
    
    return q_new, [sp_new, sm_new, uz_new, lz_new, wc_new], uz_int_2, lz_int_1
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def simulate(avg_prec, temp, et, par, p2, init_st=None, ll_temp=None, 
             q_0=DEF_q0, extra_out=False):
    '''
    ========
    Simulate
    ========

    Run the HBV model for the number of steps (n) in precipitation. The
    resluts are (n+1) simulation of discharge as the model calculates step n+1

    
    Parameters
    ----------
    avg_prec : array_like [n]
        Average precipitation [mm/h]
    temp : array_like [n]
        Average temperature [C]
    et : array_like [n]
        Potential Evapotranspiration [mm/h]
    par : array_like [18]
        Parameter vector, set up as:
        [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1, 
        alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
    p2 : array_like [2]
        Problem parameter vector setup as:
        [tfac, area]
    init_st : array_like [5], optional
        Initial model states, [sp, sm, uz, lz, wc]. If unspecified, 
        [0.0, 30.0, 30.0, 30.0, 0.0] mm
    ll_temp : array_like [n], optional
        Long term average temptearature. If unspecified, calculated from temp.
    q_0 : float, optional
        Initial discharge value. If unspecified set to 10.0
    

    Returns
    -------
    q_sim : array_like [n]
        Discharge for the n time steps of the precipitation vector [m3/s]
    st : array_like [n, 5]
        Model states for the complete time series [mm]
    '''

    if init_st is None:
        st = [DEF_ST, ]
    else:
        st = [init_st,]

    if ll_temp is None:
        ll_temp = [np.mean(temp), ] * len(avg_prec)

#    q_sim = [q_0, ]
    q_sim = [ ]
    
    #print(st)                0  1  2  3  4  5
    uz_int_2 = [st[0][2], ] #[sp,sm,uz,lz,wc,LA]
    lz_int_1 = [st[0][3], ]
    
    
    for i in range(len(avg_prec)):
#    for i in xrange(4):
        v = [avg_prec[i], temp[i], et[i], ll_temp[i]]
        q_out, st_out, uz_int_2_out, lz_int_1_out = _step_run(par, p2, v, st[i])
        q_sim.append(q_out)
        st.append(st_out)
        uz_int_2.append(uz_int_2_out) # upper zone - perc
        lz_int_1.append(lz_int_1_out) # lower zone + perc
    
#    if len(p2) > 2:  # Forcing maxbas to be predefined
#        maxbas = p2[2]  
#    elif len(par) > 18:  # Putting maxbas as parameter to be optimised
#        maxbas = par[18]
    maxbas = par[-1]
#    else:
#        maxbas = 1
    
    q_tr = _routing(np.array(q_sim), maxbas)
    
    if extra_out:
        return q_tr, st, uz_int_2, lz_int_1
    
    else:
        return q_tr, st
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def simulate_new_model(avg_prec, temp, et, par, p2, init_st=None, ll_temp=None, 
             q_0=DEF_q0, extra_out=False):
    '''
    ========
    Simulate
    ========

    Run the HBV model for the number of steps (n) in precipitation. The
    resluts are (n+1) simulation of discharge as the model calculates step n+1

    
    Parameters
    ----------
    avg_prec : array_like [n]
        Average precipitation [mm/h]
    temp : array_like [n]
        Average temperature [C]
    et : array_like [n]
        Potential Evapotranspiration [mm/h]
    par : array_like [18]
        Parameter vector, set up as:
        [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1, 
        alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
    p2 : array_like [2]
        Problem parameter vector setup as:
        [tfac, area]
    init_st : array_like [5], optional
        Initial model states, [sp, sm, uz, lz, wc]. If unspecified, 
        [0.0, 30.0, 30.0, 30.0, 0.0] mm
    ll_temp : array_like [n], optional
        Long term average temptearature. If unspecified, calculated from temp.
    q_0 : float, optional
        Initial discharge value. If unspecified set to 10.0
    

    Returns
    -------
    q_sim : array_like [n]
        Discharge for the n time steps of the precipitation vector [m3/s]
    st : array_like [n, 5]
        Model states for the complete time series [mm]
    '''

    if init_st is None:
        st = [DEF_ST, ]
    else:
        st = [init_st,]

    if ll_temp is None:
        ll_temp = [np.mean(temp), ] * len(avg_prec)

#    q_sim = [q_0, ]
    q_sim = [ ]
    
    #print(st)                0  1  2  3  4  5
    uz_int_2 = [st[0][2], ] #[sp,sm,uz,lz,wc,LA]
    lz_int_1 = [st[0][3], ]
    
    
    for i in range(len(avg_prec)):
#    for i in xrange(4):
        v = [avg_prec[i], temp[i], et[i], ll_temp[i]]
        q_out, st_out, uz_int_2_out, lz_int_1_out = _step_run(par, p2, v, st[i])
        q_sim.append(q_out)
        st.append(st_out)
        uz_int_2.append(uz_int_2_out) # upper zone - perc
        lz_int_1.append(lz_int_1_out) # lower zone + perc
    
    if extra_out:
        return q_sim, st, uz_int_2, lz_int_1
    
    else:
        return q_sim, st
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _add_mask(var, dem=None, mask=None, no_val=None):
    '''
    Put a mask in the spatially distributed values
    
    Parameters
    ----------
    var : nd_array
        Matrix with values to be masked
    cut_dem : gdal_dataset
        Instance of the gdal raster of the catchment to be cutted with. DEM 
        overrides the mask_vals and no_val
    mask_vals : nd_array
        Mask with the no_val data
    no_val : float
        value to be defined as no_val. Will mask anything is not this value
    
    Returns
    -------
    var : nd_array
        Array with masked values 
    '''
    
    if dem is not None:
        mask, no_val = _get_mask(dem)
    
    # Replace the no_data value
    assert var.shape == mask.shape, 'Mask and data do not have the same shape'
    var[mask == no_val] = no_val
    
    return var
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _get_mask(dem):
    """
    _get_mask(dem)
    to create a mask by knowing the stored value inside novalue cells 
    Inputs:
        1- flow path lenth raster
    Outputs:
        1- mask:array with all the values in the flow path length raster
        2- no_val: value stored in novalue cells
    """
    no_val = np.float32(dem.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
    mask = dem.ReadAsArray() # read all values
    return mask, no_val
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def _get_targets(dem):
    '''
    Returns the centres of the interpolation targets
    
    Parameters
    ----------
    dem : gdal_Dataset
        Get the data from the gdal datasetof the DEM
    
    Returns
    -------
    
    coords : nd_array [nxm - nan, 2]
        Array with a list of the coordinates to be interpolated, without the Nan
    
    mat_range : nd_array [n, m]
        Array with all the centres of cells in the domain of the DEM (rectangular)
    
    '''
    # Getting data for the whole grid
    x_init, xx_span, xy_span, y_init, yy_span, yx_span = dem.GetGeoTransform()
    shape_dem = dem.ReadAsArray().shape
    
    # Getting data of the mask
    no_val = dem.GetRasterBand(1).GetNoDataValue()
    mask = dem.ReadAsArray()
    
    # Adding 0.5 to get the centre
    x = np.array([x_init + xx_span*(i+0.5) for i in range(shape_dem[0])])
    y = np.array([y_init + yy_span*(i+0.5) for i in range(shape_dem[1])])
    #mat_range = np.array([[(xi, yi) for xi in x] for yi in y])
    mat_range = [[(xi, yi) for xi in x] for yi in y]
    
    # applying the mask
    coords = []
    for i in range(len(x)):
        for j in range(len(y)):
            if mask[j, i] != no_val:
                coords.append(mat_range[j][i])
                #mat_range[j, i, :] = [np.nan, np.nan]

    return np.array(coords), np.array(mat_range)
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def Dist_HBV2(lakecell,q_lake,DEM,flow_acc,flow_acc_plan, sp_prec, sp_et, sp_temp, sp_pars, p2, init_st=None, 
                ll_temp=None, q_0=None):
    '''
    Make spatially distributed HBV in the SM and UZ
    interacting cells 
    '''
    
    n_steps = sp_prec.shape[0] + 1 # no of time steps =length of time series +1
    # intiialise vector of nans to fill states
    dummy_states = np.empty([n_steps, 5]) # [sp,sm,uz,lz,wc]
    dummy_states[:] = np.nan
    
    # Get the mask
    mask, no_val = _get_mask(DEM)
    x_ext, y_ext = mask.shape # shape of the fpl raster (rows, columns)-------------- rows are x and columns are y
    #    y_ext, x_ext = mask.shape # shape of the fpl raster (rows, columns)------------ should change rows are y and columns are x
    
    # Get deltas of pixel
    geo_trans = DEM.GetGeoTransform() # get the coordinates of the top left corner and cell size [x,dx,y,dy]
    dx = np.abs(geo_trans[1])/1000.0  # dx in Km
    dy = np.abs(geo_trans[-1])/1000.0  # dy in Km
    px_area = dx*dy  # area of the cell
    
    # Enumerate the total number of pixels in the catchment
    tot_elem = np.sum(np.sum([[1 for elem in mask_i if elem != no_val] for mask_i in mask])) # get row by row and search [mask_i for mask_i in mask]
    
    # total pixel area
    px_tot_area = tot_elem*px_area # total area of pixels 
    
    # Get number of non-value data
    #%%
    st = []  # Spatially distributed states
    q_lz = []
    q_uz = []
    
    fff=0 # counter
    #------------------------------------------------------------------------------
    for x in range(x_ext): # no of rows
        st_i = []
        q_lzi = []
        q_uzi = []
    #        q_out_i = []
    # run all cells in one row ----------------------------------------------------
        for y in range(y_ext): # no of columns
            if mask [x, y] != no_val:  # only for cells in the domain
                # Calculate the states per cell
                # TODO optimise for multiprocessing these loops   
                fff+=1
                _, _st, _uzg, _lzg = simulate_new_model(avg_prec = sp_prec[:, x, y], 
                                              temp = sp_temp[:, x, y], 
                                              et = sp_et[:, x, y], 
                                              par = sp_pars[x, y, :], 
                                              p2 = p2, 
                                              init_st = init_st, 
                                              ll_temp = None, 
                                              q_0 = q_0,
                                              extra_out = True)
    #               # append column after column in the same row -----------------
                st_i.append(np.array(_st))
                #calculate upper zone Q = K1*(LZ_int_1)
                q_lz_temp=np.array(sp_pars[x, y, 6])*_lzg
                q_lzi.append(q_lz_temp)
                # calculate lower zone Q = k*(UZ_int_3)**(1+alpha)
                q_uz_temp = np.array(sp_pars[x, y, 5])*(np.power(_uzg, (1.0 + sp_pars[x, y, 7])))
                q_uzi.append(q_uz_temp)
                
    #                print("total = "+str(fff)+"/"+str(tot_elem)+" cell, row= "+str(x+1)+" column= "+str(y+1) )
            else: # if the cell is novalue-------------------------------------
                # Fill the empty cells with a nan vector
                st_i.append(dummy_states) # fill all states(5 states) for all time steps = nan
                q_lzi.append(dummy_states[:,0]) # q lower zone =nan  for all time steps = nan
                q_uzi.append(dummy_states[:,0]) # q upper zone =nan  for all time steps = nan
    # store row by row-------- ---------------------------------------------------- 
    #        st.append(st_i) # state variables 
        st.append(st_i) # state variables 
        q_lz.append(np.array(q_lzi)) # lower zone discharge mm/timestep
        q_uz.append(np.array(q_uzi)) # upper zone routed discharge mm/timestep
    #------------------------------------------------------------------------------            
    # convert to arrays 
    st = np.array(st)
    q_lz = np.array(q_lz)
    q_uz = np.array(q_uz)
    #%% convert quz from mm/time step to m3/sec
    area_coef=p2[1]/px_tot_area
    q_uz=q_uz*px_area*area_coef/(p2[0]*3.6)
    
    no_cells=list(set([flow_acc_plan[i,j] for i in range(x_ext) for j in range(y_ext) if not np.isnan(flow_acc_plan[i,j])]))
#    no_cells=list(set([int(flow_acc_plan[i,j]) for i in range(x_ext) for j in range(y_ext) if flow_acc_plan[i,j] != no_val]))
    no_cells.sort()

    #%% routing lake discharge with DS cell k & x and adding to cell Q
    q_lake=Routing.muskingum(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
    q_lake=np.append(q_lake,q_lake[-1])
    # both lake & Quz are in m3/s
    #new
    q_uz[lakecell[0],lakecell[1],:]=q_uz[lakecell[0],lakecell[1],:]+q_lake
    #%% cells at the divider
    q_uz_routed=np.zeros_like(q_uz)*np.nan
    # for all cell with 0 flow acc put the q_uz
    for x in range(x_ext): # no of rows
        for y in range(y_ext): # no of columns
            if mask [x, y] != no_val and flow_acc_plan[x, y]==0: 
                q_uz_routed[x,y,:]=q_uz[x,y,:]        
    #%% new
    for j in range(1,len(no_cells)): #2):#
        for x in range(x_ext): # no of rows
            for y in range(y_ext): # no of columns
                    # check from total flow accumulation 
                    if mask [x, y] != no_val and flow_acc_plan[x, y]==no_cells[j]:
#                        print(no_cells[j])
                        q_r=np.zeros(n_steps)
                        for i in range(len(flow_acc[str(x)+","+str(y)])): #  no_cells[j]
                            # bring the indexes of the us cell
                            x_ind=flow_acc[str(x)+","+str(y)][i][0]
                            y_ind=flow_acc[str(x)+","+str(y)][i][1]
                            # sum the Q of the US cells (already routed for its cell)
                             # route first with there own k & xthen sum
                            q_r=q_r+Routing.muskingum(q_uz_routed[x_ind,y_ind,:],q_uz_routed[x_ind,y_ind,0],sp_pars[x_ind,y_ind,10],sp_pars[x_ind,y_ind,11],p2[0]) 
#                        q=q_r
                         # add the routed upstream flows to the current Quz in the cell
                        q_uz_routed[x,y,:]=q_uz[x,y,:]+q_r
    #%% check if the max flow _acc is at the outlet
#    if tot_elem != np.nanmax(flow_acc_plan):
#        raise ("flow accumulation plan is not correct")
    # outlet is the cell that has the max flow_acc
    outlet=np.where(flow_acc_plan==np.nanmax(flow_acc_plan)) #np.nanmax(flow_acc_plan)
    outletx=outlet[0][0]
    outlety=outlet[1][0]              
    #%%
    q_lz = np.array([np.nanmean(q_lz[:,:,i]) for i in range(n_steps)]) # average of all cells (not routed mm/timestep)
    # convert Qlz to m3/sec 
    q_lz = q_lz* p2[1]/ (p2[0]*3.6) # generation
    
    q_out = q_lz + q_uz_routed[outletx,outlety,:]    

    return q_out, st, q_uz_routed, q_lz, q_uz
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""