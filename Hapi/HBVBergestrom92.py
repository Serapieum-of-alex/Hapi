"""
======
Lumped Conceptual HBV model
======

HBV is lumped conceptual model consists of precipitation, snow melt,
soil moisture and response subroutine to convert precipitation into o runoff,
where state variables are updated each time step to represent a specific
hydrologic behaviour the catchment

This version was edited based on a Master Thesis on "Spatio-temporal simulation
of catchment response based on dynamic weighting of hydrological models" on april 2018

- Model inputs are Precipitation, evapotranspiration and temperature, initial
    state variables, and initial discharge.
- Model output is Qalculated dicharge at time t+1
- Model equations are solved using explicit scheme 
- model structure uses 18 parameters if the catchment has snow
    [tt, rfcf, sfcf, cfmax, cwh, cfr, fc, beta, e_corr, lp, c_flux, k, k1,
    alpha, perc]
    
    otherwise it uses 10 parameters
    [rfcf, fc, beta, etf, lp, c_flux, k, k1, alpha, perc]

this HBV is based on Bergstrom, 1992 two reservoirs with three linear responses
surface runoff, interflow and baseflow
"""
# libraries
import numpy as np

DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0]
DEF_q0 = 0

# Get random parameter set
#def get_random_pars():
#    return np.random.uniform(P_LB, P_UB)


def Precipitation(prec, temp, tt, rfcf, sfcf):
    """
    ========================================================
    Precipitation (temp, tt, prec, rfcf, sfcf)
    ========================================================

    Precipitaiton routine of the HBV96 model.

    If temperature is lower than TT [degree C], all the precipitation is considered as
    snow. If the temperature is higher than tt, all the precipitation is
    considered as rainfall.

    Parameters
    ----------
    temp : float
        Measured temperature [C]
    tt : float
        Lower temperature treshold [C]
    prec : float 
        Precipitation [mm]
    rfcf : float
        Rainfall corrector factor
    sfcf : float
        Snowfall corrector factor

    Returns
    -------
    rf : float
        Rainfall [mm]
    sf : float
        Snowfall [mm]
    """

    if temp <= tt:  # if temp <= lower temp threshold 
        rf = 0.0         #no rainfall all the precipitation will convert into snowfall
        sf = prec*sfcf

    else : #temp >= tt: # if temp > upper threshold 
        rf = prec*rfcf # no snowfall all the precipitation becomes rainfall 
        sf = 0.0

    return rf, sf


def Snow(temp, rf, sf, wc_old, sp_old, tt, cfmax, cfr, cwh):
    """
    ========================================================
        Snow(cfmax, temp, tt, cfr, cwh, rf, sf, wc_old, sp_old)
    ========================================================
    Snow routine.
    
    The snow pack consists of two states: Water Content (wc) and Snow Pack 
    (sp). The first corresponds to the liquid part of the water in the snow,
    while the latter corresponds to the solid part. If the temperature is 
    higher than the melting point, the snow pack will melt and the solid snow
    will become liquid. In the opposite case, the liquid part of the snow will
    refreeze, and turn into solid. The water that cannot be stored by the solid
    part of the snow pack will drain into the soil as part of infiltration.
    
    All precipitation simulated to be snow, i.e. falling when the temperature
    is bellow TT, is multiplied by a snowfall correction factor, SFCF [-].
    
    Snowmelt is calculated with the degree-day method (cfmax). Meltwater and
    rainfall is retained within the snowpack until it exceeds a certain
    fraction, CWH [%] of the water equivalent of the snow
    
    Liquid water within the snowpack refreezes using cfr

    Parameters
    ----------
    cfmax : float 
        Day degree factor
    temp : float 
        Temperature [C]
    tt : float 
        Temperature treshold for Melting [C]
    cfr : float 
        Refreezing factor
    cwh : float 
        Capacity for water holding in snow pack [%]
    rf : float 
        Rainfall [mm]
    sf : float 
        Snowfall [mm]
    wc_old : float 
        Water content in previous state [mm]
    sp_old : float 
        snow pack in previous state [mm]

    Returns
    -------
    in : float 
        Infiltration [mm]
    wc_new : float 
        Liquid Water content in the snow[mm]
    sp_new : float 
        Snowpack in posterior state [mm]
    """

    if temp > tt:# if temp > melting threshold
        # then either some snow will melt or the entire snow will melt 
        if cfmax*(temp-tt) < sp_old+sf: #if amount of melted snow < the entire existing snow (previous amount+new)
            melt = cfmax*(temp-tt)
        else:                             #if amount of melted snow > the entire existing snow (previous amount+new)
            melt = sp_old+sf           # then the entire existing snow will melt (old snow pack + the current snowfall)

        sp_new = sp_old + sf - melt
        wc_int = wc_old + melt + rf

    else: # if temp < melting threshold
        #then either some water will freeze or all the water willfreeze
        if cfr*cfmax*(tt-temp) < wc_old+rf: #then either some water will freeze or all the water willfreeze
            refr = cfr*cfmax*(tt - temp)  #cfmax*(ttm-temp) is the rate of melting of snow while cfr*cfmax*(ttm-temp) is the rate of freeze of melted water  (rate of freezing > rate of melting) 
        else:                               # if the amount of frozen water > entire water available
            refr = wc_old + rf

        sp_new = sp_old + sf + refr
        wc_int = wc_old - refr + rf
    
    # 
    if wc_int > cwh*sp_new: # if water content > holding water capacity of the snow
        inf = wc_int-cwh*sp_new  #water content  will infiltrate
        wc_new = cwh*sp_new # and the capacity of snow of holding water will retained
    else:           # if water content < holding water capacity of the snow
        inf = 0.0            # no infiltration
        wc_new = wc_int

    return inf, wc_new, sp_new



def Soil(temp, inf, ep, sm_old, uz_old, tm, fc, beta, e_corr, lp, c_flux):
    """
    ============================================================
        Soil(fc, beta, temp, tm, e_corr, lp, c_flux, inf, ep, sm_old, uz_old)
    ============================================================
    
    Soil routine of the HBV-96 model.
    
    The model checks for the amount of water that can infiltrate the soil, 
    coming from the liquid precipitation and the snow pack melting. A part of 
    the water will be stored as soil moisture, while other will become runoff, 
    and routed to the upper zone tank.
    
    -Actual evaporation from the soil box equals the potential evaporation if
    SM/FC is above LP [%] while a linear reduction is used when SM/FC is below
    LP.    
    -Groundwater recharge (r) is added to the upper groundwater box (UZ [mm])
    -PERC [mm d-1]defines the maximum percolation rate from the upper to the
    lower groundwater box (LZ [mm])

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
    """

#    qdr = max(sm_old + inf - fc, 0)  # direct run off as soil moisture exceeded the field capacity

#    inf = inf - qdr
    
    # recharge to the upper zone
    r = ((sm_old/fc)** beta) * inf
    
    # Adjusted potential evapotranspiration
    ep_int = (1.0 + (temp - tm))*e_corr*ep
    
    ea = min(ep_int, (sm_old/(lp*fc))*ep_int)
    
    # capilary rise
    cf = c_flux*((fc - sm_old)/fc) 
    
    # if capilary rise is more than what is available take all the available and leave it empty
    
    if uz_old + r < cf: 
        cf= uz_old + r
        uz_int_1=0
    else:
        uz_int_1 = uz_old + r - cf
    
    sm_new = max(sm_old + inf - r + cf - ea, 0)
    
    return sm_new, uz_int_1


def Response(tfac, lz_old, uz_int_1, perc, k, k1, k2, uzl):
    """
    ============================================================
        Response(tfac, perc, k, k1, k2, uzl, area, lz_old, uz_int_1)
    ============================================================
    The response routine of the HBV-96 model.
    
    The response routine is in charge of transforming the current values of 
    upper and lower zone into discharge. This routine also controls the 
    recharge of the lower zone tank (baseflow). The transformation of units 
    also occurs in this point.
    
    - PERC [mm d-1]defines the maximum percolation rate from the upper to the
    lower groundwater box (SLZ [mm])
    
    - Runoff from the groundwater boxes is computed as the sum of two or three
    linear outflow equatins depending on whether UZ is above a threshold value,
    UZL [mm], or not
    
    Parameters
    ----------
        tfac : float
            Number of hours in the time step
        perc : float
            Percolation value [mm\hr]
        k : float
            direct runoff recession coefficient
        k1 : float 
            Upper zone recession coefficient
        k2 : float 
            Lower zone recession coefficient
        uzl: float
            Upper zone threshold value
        area : float
            Catchment area [Km2]
        lz_old : float 
            Previous lower zone value [mm]
        uz_int_1 : float 
            Previous upper zone value before percolation [mm]
        
    Returns:
    ----------
        1-q_uz:[float]
            upper zone discharge (mm/timestep) for both surface runoff & interflow
        2-q_2:[float]
            Lower zone discharge (mm/timestep).
    """
    # upper zone 
    # if perc > Quz then perc = Quz and Quz = 0 if not perc = value and Quz= Quz-perc so take the min 
    uz_int_2 = np.max([uz_int_1 - perc, 0.0])
    
    # surface runoff
    q_0 = k*np.max([uz_int_2-uzl,0])
    
    # Interflow
    q_1 = k1*(uz_int_2)
    
    # as K & k1 are a very small values (0.005) this condition will never happen 
    if q_0+q_1 > uz_int_2: # if q_0 =30 and UZ=20
        q_0= uz_int_2*0.67  # q_0=20
        q_1= uz_int_2*0.33
        
    uz_new = uz_int_2 - (q_0+q_1) 
    
    # lower zone tank
    # if the percolation > upper zone Q all the Quz will percolate
    lz_int_1 = lz_old + np.min([perc, uz_int_1])
    
    q_2 = k2*lz_int_1
    
    if q_2 > lz_int_1:
        q_2=lz_int_1
    
    lz_new = lz_int_1 - (q_2)
    
    q_uz=q_0+q_1
#    q_new = area*(q_0 + q_1)/(3.6*tfac)  # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
                                                    # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25

#    return q_new, uz_new, lz_new, uz_int_2, lz_int_1
    return q_uz, q_2, uz_new, lz_new #,uz_int_2, lz_int_1


def Tf(maxbas):
    """
     ====================================================
         Tf(maxbas)
     ====================================================
    Transfer function weight generator 
    
    """
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


def Routing(q, maxbas=1):
    """
    ==================================================
        Routing(q, maxbas=1)
    ==================================================
    This function implements the transfer function using a triangular 
    function
    """
    assert maxbas >= 1, 'Maxbas value has to be larger than 1'
    # Get integer part of maxbas
#    maxbas = int(maxbas)
    maxbas = int(round(maxbas,0))
    
    # get the weights
    w = Tf(maxbas)
    
    # rout the discharge signal
    q_r = np.zeros_like(q, dtype='float64')
    q_temp = q
    for w_i in w:
        q_r += q_temp*w_i
        q_temp = np.insert(q_temp, 0, 0.0)[:-1]

    return q_r


def StepRun(p, p2, v, St, snow=0):
    """
    ============================================================
        StepRun(p, p2, v, St, snow=0)
    ============================================================
    
    Makes the calculation of next step of discharge and states
    
    Parameters
    ----------
    p : array_like [18]
        Parameter vector, set up as:
        [tt, rfcf, sfcf, cfmax, cfr, cwh, fc, ecorr, etf, lp, k, k1, 
        alpha, beta, cwh, cfr, c_flux, perc, ]
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
    """
    ## Parse of parameters from input vector to model
    #picipitation function
    if snow==1:
        assert len(p) == 16, "current version of HBV (with snow) takes 18 parameter you have entered "+str(len(p))
        tt = p[0]
        rfcf = p[1] 
        sfcf = p[2]  
        # snow function
        cfmax = p[3] 
        cwh = p[4]    
        cfr = p[5]
        #soil function
        fc = p[6]       
        beta = p[7]     
        e_corr =[8]
        lp =p[9]        
        c_flux =p[10]
        # response function
        k = p[11]
        k1 = p[12]
        k2 = p[13]
        uzl = p[14]
        perc = p[15]
        
    elif snow == 0:
        assert len(p) >= 11, "current version of HBV (without snow) takes 11 parameter you have entered "+str(len(p))
        tt = 2.0     # very low but it does not matter as temp is 25 so it is greater than 2
        rfcf =p[0]    # 1.0 #p[16] # all precipitation becomes rainfall
        sfcf = 0.00001  # there is no snow
        # snow function
        cfmax = 0.00001  # as there is no melting  and sp+sf=zero all the time so it doesn't matter the value of cfmax
        cwh = 0.00001    # as sp is always zero it doesn't matter all wc will go as inf 
        cfr = 0.000001   # as temp > ttm all the time so it doesn't matter the value of cfr but put it zero
        #soil function
        fc = p[1]       
        beta = p[2]     
        e_corr =p[3]
        lp =p[4]        
        c_flux =p[5]   
        # response function
        k = p[6]
        k1 = p[7]
        k2 = p[8]
        uzl = p[9]
        perc = p[10]
        
    ## Non optimisable parameters
    tfac = p2[0] 
    area = p2[1] 

    ## Parse of Inputs
    prec = v[0] # Precipitation [mm]
    temp = v[1] # Temperature [C]
    ep = v[2] # Long terms (monthly) Evapotranspiration [mm]
    tm = v[3] #Long term (monthly) average temperature [C]

    ## Parse of states
    sp_old = St[0]
    sm_old = St[1]
    uz_old = St[2]
    lz_old = St[3]
    wc_old = St[4]

    rf, sf = Precipitation(prec, temp, tt, rfcf, sfcf)
    
    inf, wc_new, sp_new = Snow(temp, rf, sf, wc_old, sp_old,
                               tt, cfmax, cfr, cwh)
    
    sm_new, uz_int_1 = Soil(temp, inf, ep, sm_old, uz_old, tm,
                            fc, beta, e_corr, lp, c_flux)

    q_uz, q_lz, uz_new, lz_new = Response(tfac, lz_old, uz_int_1,
                                          perc, k, k1, k2, uzl)
   
#    return q_new, [sp_new, sm_new, uz_new, lz_new, wc_new], uz_int_2, lz_int_1
    return q_uz, q_lz, [sp_new, sm_new, uz_new, lz_new, wc_new]


def Simulate(prec, temp, et, par, p2, init_st=None, ll_temp=None, 
             q_init=None, snow=0):
    """    
    ================================================================
        Simulate(prec, temp, et, par, p2, init_st=None, ll_temp=None, q_init=None, snow=0):
    ================================================================
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
    q_init : float, optional
        Initial discharge value. If unspecified set to 10.0
    

    Returns
    -------
    q_sim : array_like [n]
        Discharge for the n time steps of the precipitation vector [m3/s]
    st : array_like [n, 5]
        Model states for the complete time series [mm]
    """
    ### inputs validation
    # data type
    assert len(init_st) == 5, "state variables are 5 and the given initial values are "+str(len(init_st))
    assert type(p2) == list, " p2 should be of type list"
    assert len(p2) == 2, "p2 should contains tfac and catchment area"
    assert snow == 0 or snow == 1, " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"

    if init_st is None:#   0  1  2  3  4  5
        st = [DEF_ST, ]  #[sp,sm,uz,lz,wc,LA]
    else:
        st = [init_st,]

    if ll_temp is None:
        ll_temp = [np.mean(temp), ] * len(prec)
        
    ### initial runoff 
    # calculate the runoff for the first time step
    if q_init == None:
        if snow == 1:
            # upper zone
            q_0=[par[11]*np.max([st[0][2] - par[14],0]), ]
            q_1=[par[12]*((st[0][2])), ]
            q_uz = [q_0[0]+q_1[0],]
            # lower zone 
            q_lz=[par[13]*st[0][3], ]
        else:
            # upper zone
            q_0=[par[6]*np.max([st[0][2] - par[9],0]), ]
            q_1=[par[7]*((st[0][2])), ]
            q_uz = [q_0[0]+q_1[0],]
            # lower zone 
            q_lz=[par[8]*st[0][3], ]
    else: # if initial runoff value is given distribute it evenlt between upper and lower responses 
        q_uz = [q_init/2, ]
        q_lz = [q_init/2, ]
    
    
    for i in range(len(prec)):
        v = [prec[i], temp[i], et[i], ll_temp[i]]
        q_uzi, q_lzi, st_out = StepRun(par, p2, v, st[i], snow=0)
        q_uz.append(q_uzi)
        q_lz.append(q_lzi)
        st.append(st_out)
    
    return np.float32(q_uz), np.float32(q_lz), np.float32(st)