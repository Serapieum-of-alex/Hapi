"""
======
HBV-96
======
Lumped hydrological model.
This is the HBV-96 implementation by Juan Chacon at IHE-Delft, NL. This code
implements the HBV-96 version, as described in Lindstrom et al (1997)
https://doi.org/10.1016/S0022-1694(97)00041-3
@author: Juan Carlos Chacon-Hurtado (jc.chaconh@gmail.com)                                  
Version
-------
03-05-2017 - V_0.0 - First implementation
"""
from __future__ import division, print_function
import numpy as np
#import scipy.optimize as opt
from scipy.interpolate import InterpolatedUnivariateSpline as interp11
import sklearn.metrics as error

# HBV base model parameters
# 19 parameter (18+maxbas)
## parameters lower limit
#P_LB = [50.0, #1-fc
#        0.01, #2-beta
#        0.6, #3-ecorr
#        0.001, #4-etf
#        0.001, #5-lp
#        0.0, #6-c_flux
#        0.001, #7-k [h^-1] upper zone
#        0.0001, #8-k1 lower zone
#        0.01, #9-alpha
#        0.01, #10-perc mm/h
#        0.01, #11-c_le 
#        2] # 12-Maxbas
   
# parameters upper limit
#P_UB = [500.0, #1-fc
#        5.0, #2-beta
#        3, #3-ecorr
#        5.0, #4-etf
#        1.0, #5-lp
#        1.0, #6-c_flux - 2mm/day
#        0.1, #7-k upper zone
#        0.1, #8-k1 lower zone
#        1.0, #9-alpha
#        2.0, #10-perc mm/hr
#        1.0, #11-c_le
#        15] # 12-maxbas
              
# initial values for state variables
          #[sp, sm, uz, lz, wc]
DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0,10.163*10**9]
#initial value for discarge
DEF_q0 =2.3    #10.0

# Get random parameter set
#def get_random_pars():
#    # this function returns a random value for the parameters between upper and lower bound 
#    return np.random.uniform(P_LB, P_UB)

def _precipitation(temp, ltt, utt, prec, rfcf, sfcf, tfac,pcorr):
    '''
    ==============
    Precipitation
    ==============
    inputs:
        1-precipitation
        2- Temperature
    outputs:
        1- rainfall
        2- snowfall
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

    if temp <= ltt:      # if temp <= lower temp threshold 
        _rf = 0.0        #no rainfall all the precipitation will convert into snowfall
        _sf = prec*sfcf 

    elif temp >= utt:    # if temp > upper threshold 
        _rf = prec*rfcf  # no snowfall all the precipitation becomes rainfall 
        _sf = 0.0

    else:                # if  ltt< temp < utt
        _rf = ((temp-ltt)/(utt-ltt)) * prec * rfcf 
        _sf = (1.0-((temp-ltt)/(utt-ltt))) * prec * sfcf
    
    _rf=_rf*pcorr

    return _rf, _sf


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
        Day degree factor   # melting factor
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
        water-holding capacity of snow (meltwater is retained in snowpack until it exceeds the WHC)
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

    if temp > ttm: # if temp > melting threshold
        # then either some snow will melt or the entire snow will melt 
        if cfmax*(temp-ttm) < sp_old+_sf:  #if amount of melted snow < the entire existing snow (previous amount+new)
            _melt = cfmax*(temp-ttm)
        else:                              #if amount of melted snow > the entire existing snow (previous amount+new)
            _melt = sp_old+_sf             # then the entire existing snow will melt (old snow pack + the current snowfall)

        _sp_new = sp_old + _sf - _melt
        _wc_int = wc_old + _melt + _rf

    else:         # if temp < melting threshold
        #then either some water will freeze or all the water willfreeze
        if cfr*cfmax*(ttm-temp) < wc_old+_rf: # if the amount of frozen water < entire water available
            _refr = cfr*cfmax*(ttm - temp)    #cfmax*(ttm-temp) is the rate of melting of snow while cfr*cfmax*(ttm-temp) is the rate of freeze of melted water  (rate of freezing > rate of melting) 
        else:                                # if the amount of frozen water > entire water available
            _refr = wc_old + _rf

        _sp_new = sp_old + _sf + _refr
        _wc_int = wc_old - _refr + _rf

    if _wc_int > cwh*_sp_new:   # if water content > holding water capacity of the snow
        _in = _wc_int-cwh*_sp_new   #water content  will infiltrate
        _wc_new = cwh*_sp_new       # and the capacity of snow of holding water will retained
    else:                      # if water content < holding water capacity of the snow
        _in = 0.0                   # no infiltration
        _wc_new = _wc_int

    return _in, _wc_new, _sp_new



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
        Controls the contribution of the increase in the soil moisture or
        to the response function
    etf : float 
        Total potential evapotranspiration
    temp : float 
        Temperature
    tm : float 
        Average long term temperature
    e_corr : float 
        Evapotranspiration corrector factor
    lp : float _soil 

    tfac : float 
        Time conversion factor
    c_flux : float 
        Capilar flux in the root zone
    _in : float 
        actual infiltration
    ep : float 
        Long term mean potential evapotranspiration
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
    
    variables
    -------
    EPInt  :
        Adjusted potential evapotranspiration
    '''

#    qdr = max(sm_old + inf - fc, 0)  # direct run off as soil moisture exceeded the field capacity
    qdr=0
    _in = inf - qdr
    _r = ((sm_old/fc)** beta) * _in  # recharge from soil subroutine to upper zone 
    
    _ep_int = max((1.0 + etf*(temp - tm))*e_corr*ep,0)  # Adjusted potential evapotranspiration
    
    _ea = min(_ep_int, (sm_old/(lp*fc))*_ep_int)

    _cf = c_flux*((fc - sm_old)/fc) # capilary rise
    
    # if capilary rise is more than what is available take all the available and leave it empty
    
    if uz_old + _r < _cf: 
        _cf= uz_old + _r
        uz_int_1=0
    else:
        uz_int_1 = uz_old + _r - _cf
        
#    uz_int_1 = max(uz_old + _r - _cf,0)
    
    sm_new = max(sm_old + _in - _r + _cf - _ea, 0)
    

    return sm_new, uz_int_1, qdr



def _response(tfac, perc, alpha, k, k1, area, lz_old, uz_int_1, qdr):
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
        upper zone runoff coefficient
    k : float
        Upper zone recession coefficient
        Upper zone response coefficient
    k1 : float 
        Lower zone recession coefficient
        Lowe zone response coefficient
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
    uz_int_2 = np.max([uz_int_1 - perc, 0.0]) # upper zone after percolation
    _q_0 = k*(uz_int_2**(1.0 + alpha))
    
    if _q_0 > uz_int_2: # if q_0 =30 and UZ=20
        _q_0= uz_int_2  # q_0=20 

    uz_new = uz_int_2 - (_q_0)
    
    lz_int_1 = lz_old + np.min([perc, uz_int_1]) # if the percolation > upper zone Q all the Quz will percolate
    
    _q_1 = k1*lz_int_1
    
    if _q_1 > lz_int_1:
        _q_1=lz_int_1
        
    lz_new = lz_int_1 - (_q_1)
    
    q_new = ((_q_0+_q_1)*area)/(3.6*tfac)   # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
                                                    # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
    return q_new, uz_new, lz_new


def _lake(temp,curve,tfac,rf,sf,q_new,lv_old,ltt,c_le,ep,lakeA):
    
    # lower zone
    #explicit representation of the lake where lake will be represented by a rating curve
    '''
    l_ea :lake evaporation
    c_le : lake _evaporation correction factor
    
    '''
    # lake evaporation
    if temp>= ltt :
        l_ea=c_le*ep   #the evaporation will be the potential evapotranspiration times correction factor
    else:
        l_ea=0          #Evaporation will not occur when the Temperature is below the Threshold temperature    
    
    l_ea_vol=l_ea*lakeA*1000   # evaporation volume m3/time step
    
    # evaporation on the lake
    if temp < ltt:
        l_p=sf*c_le
    else:
        l_p=rf*c_le
    
    l_p_vol= l_p*lakeA*1000   # prec # precipitation volume/ timestep
    
    q_vol=q_new*3600*tfac     # volume of inflow to the lake
    
    # storage in the lake before calculating the outflow 
    lkv1=lv_old + l_p_vol + q_vol - l_ea_vol
    # average storage for interpolation
    lkv2=(lkv1+lv_old)/2
    
    storage=curve[:,1]
    discharge=curve[:,0]
    fn=interp11(storage,discharge,k=1)
    qout=max(fn(lkv2).tolist(),0)
    
    lv_new=lkv2-(qout*3600*tfac)
    
    return qout,lv_new


def _tf(maxbas):
    ''' Transfer function weight generator 
     in a shape of a triangle 
    '''
    
    wi = []
    for x in range(1, maxbas+1): # if maxbas=3 so x=[1,2,3]
        if x <= (maxbas)/2.0:   # x <= 1.5  # half of values will form the rising limb and half falling limb
            # Growing transfer    # rising limb
            wi.append((x)/(maxbas+2.0))
        else:
            # Receding transfer    # falling limb
            wi.append(1.0 - (x+1)/(maxbas+2.0))
    
    #Normalise weights
    wi = np.array(wi)/np.sum(wi)
    return wi


def _routing(q, maxbas=1):
    """
    This function implements the transfer function using a triangular 
    function
    """
    assert maxbas >= 1, 'Maxbas value has to be larger than 1'
    # Get integer part of maxbas
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



def CalculateMaxBas(MAXBAS):
    """
    This function calculate the MAXBAS Weights based on a MAXBAX number
    The MAXBAS is a HBV parameter that controls the routing
    Example: 
    maxbasW = CalculateMaxBas(5)
    maxbasW =
    
        0.0800    0.2400    0.3600    0.2400    0.0800
    
    It is important to mention that this function allows to obtain weights
    not only for interger values but from decimals values as well.
    """
    
    yant = 0
    Total = 0 # Just to verify how far from the unit is the result
    
    TotalA = (MAXBAS * MAXBAS * np.sin(np.pi/3)) / 2
    
    
    IntPart = np.floor(MAXBAS)
    
    RealPart = MAXBAS - IntPart
    
    PeakPoint = MAXBAS%2
    
    flag = 1  # 1 = "up"  ; 2 = down
    
    if RealPart>0 : # even number 2,4,6,8,10 
        maxbasW=np.ones(int(IntPart)+1) # if even add 1
    else:            # odd number
        maxbasW=np.ones(int(IntPart))
    
    
    for x in range(int(MAXBAS)):
        
        if x < (MAXBAS / 2.0)-1:
            ynow = np.tan(np.pi/3) * (x+1); #Integral of  x dx with slope of 60 degree Equilateral triangle
            maxbasW[x] = ((ynow + yant) / 2) / TotalA # ' Area / Total Area
            
        else:     #The area here is calculated by the formlua of a trapezoidal (B1+B2)*h /2
            if flag == 1 :
                ynow = np.sin(np.pi/3) * MAXBAS;
                if PeakPoint == 0 :
                    maxbasW[x] = ((ynow + yant) / 2) / TotalA;
                else:
                    A1 = ((ynow + yant) / 2) * (MAXBAS / 2.0 - x ) / TotalA;
                    yant = ynow;
                    ynow = (MAXBAS * np.sin(np.pi/3)) - (np.tan(np.pi/3) * (x+1 - MAXBAS / 2.0));
                    A2 = ((ynow + yant) * (x +1 - MAXBAS / 2.0) / 2) / TotalA;
                    maxbasW[x] = A1 + A2;

                flag = 2
            else:
                ynow = (MAXBAS * np.sin(np.pi / 3) - np.tan(np.pi / 3) * (x+1 - MAXBAS / 2.0))#'sum of the two height in the descending part of the triangle
                maxbasW[x] = ((ynow + yant) / 2) / TotalA; #Multiplying by the height of the trapezoidal and dividing by 2
            
        Total = Total + maxbasW[x];
        yant = ynow;

    
    x = x + 1; 
    
    if RealPart > 0 :
        if np.floor(MAXBAS)== 0 :
            MAXBAS = 1
            maxbasW[x] = 1
            NumberofWeights = 1
        else:
            maxbasW[x] = (yant * (MAXBAS - (x)) / 2) / TotalA
            Total = Total + maxbasW[x]
            NumberofWeights = x
    else:
    
        NumberofWeights = x - 1;
    
    return maxbasW

def RoutingMAXBAS(Q,MAXBAS):
    """
    This function calculate the routing from a input hydrograph using 
    the MAXBAS parameter from the HBV model.
    EXAMPLE: 
    
    [Qout,maxbasW]=RoutingMAXBAS(Q,5);
    where:
    Qout = output hydrograph
    maxbasW = MAXBAS weight
    Q = input hydrograph
    5 = MAXBAS parameter value.
    """
    
    # CALCULATE MAXBAS WEIGHTS
    
    maxbasW = CalculateMaxBas(MAXBAS);
    Qw=np.ones((len(Q),len(maxbasW)))
    # Calculate the matrix discharge
    for i in range(len(Q)): # 0 to 10 
        for k in range(len(maxbasW)):  # 0 to 4
            Qw[i,k] = maxbasW[k]*Q[i]
    
    def mm(A,s):
        tot=[]
        for o in range(np.shape(A)[1]): # columns
            for t in range(np.shape(A)[0]): # rows
                tot.append(A[t,o])
        Su=tot[s:-1:s]
        return Su
    
    # Calculate routing
    j = 0
    Qout=np.ones((len(Q),1))
    
    for i in range(len(Q) ):
        if i == 0:
            Qout[i] = Qw[i,i]
        elif i < len(maxbasW)-1:
            A = Qw[0:i+1,:]
            s = len(A)-1    # len(A) is the no of rows or use int(np.shape(A)[0])
            Su=mm(A,s)
            
            Qout[i] = sum(Su[0:i+1])
        else:
            A = Qw[j:i+1,:]
            s = len(A)-1 
            Su=mm(A,s)
            Qout[i] = sum(Su)
            j = j + 1
    
#        del A ,s, Su
    
    return Qout #,maxbasW


# run all function step by step
def _step_run(p, p2, v, St,curve,lake_sim):
    '''
    ========
    Step run
    ========
    
    Makes the calculation of next step of discharge and states
    
    Parameters
    ----------
    p : array_like [10]
        Parameter vector, set up as:
        [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]
        9 fixed parameters + 9 optimized parameters + 1 parameter for the lake 
    p2 : array_like [3]
        Problem parameter vector setup as:
        [tfac, area, Lake_area]
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
    
    ## Non optimisable parameters
    #[tfac,jiboa,lake_sub,lake_area]
    tfac = p2[0] # tfac=0.25
    jiboa_area=p2[1] # AREA = 432
    lake_sub=p2[2] # area of lake subcatchment
    lakeA= p2[3] # area of the lake 
#    ilake=lakeA/area  # percentage of the lake area to catchment area

    ## Parse of Inputs
    avg_prec = v[0] # Precipitation [mm]
    temp = v[1] # Temperature [C]
    ep = v[2]  
    tm = v[3] #Long term (monthly) average temperature [C]

    ## Parse of states
    sp_old = St[0]
    sm_old = St[1]
    uz_old = St[2]
    lz_old = St[3]
    wc_old = St[4]
    
    if lake_sim:
        area=lake_sub
        c_le = p[9]
        lv_old = St[5]    
        pcorr=p[10]
    else:
        area=jiboa_area
        pcorr=p[9]
    
#    pcorr=p[10]

    rf, sf = _precipitation(temp, ltt, utt, avg_prec, rfcf, sfcf, tfac, pcorr)
    inf, wc_new, sp_new = _snow(cfmax, tfac, temp, ttm, cfr, cwh, rf, sf,
                               wc_old, sp_old)
    sm_new, uz_int_1, qdr = _soil(fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux,
                            inf, ep, sm_old, uz_old)
    q_new, uz_new, lz_new = _response(tfac, perc, alpha, k, k1, area, lz_old,
                                    uz_int_1, qdr)
    
    # if lake_sim is true it will enter the function of the lake
    if lake_sim:
        qout, lv_new = _lake(temp,curve,tfac,rf,sf,q_new,lv_old,ltt,c_le,ep,lakeA)
    else:    # if lake_sim is false it will enter the function of the lake
        qout=q_new
        lv_new=0
    
    return qout, [sp_new, sm_new, uz_new, lz_new, wc_new, lv_new]


# run the step by step function in for loop & the routing function

def simulate(avg_prec, temp, et, par, p2, curve,q_0=DEF_q0,init_st=None, 
              ll_temp=None,lake_sim=False):
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
    par : array_like [10]
        Parameter vector, set up as:
        [fc, beta, e_corr, etf, lp, c_flux, k, k1, alpha, perc]
    p2 : array_like [3]
        Problem parameter vector setup as:
        [tfac, area, Lake_area]
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

    if init_st is None:  #If unspecified, [0.0, 30.0, 30.0, 30.0, 0.0] mm
        st = [DEF_ST, ] # if not given take the default
    else:
        st=[init_st,]   # if given take it 

    if ll_temp is None: #If Long term average temptearature unspecified, calculated from temp
        ll_temp = [np.mean(temp), ] * len(avg_prec)
    

#    q_sim = [q_0, ]
    q_sim = [ ]

# run the step by step function
#    for i in xrange(len(avg_prec)):
    for i in range(len(avg_prec)):
        v = [avg_prec[i], temp[i], et[i], ll_temp[i]]
        q_out, st_out = _step_run(par, p2, v, st[i],curve,lake_sim)
        q_sim.append(q_out)
        st.append(st_out)

#run the routing function
    # search  for the maxbas parameter in the parameter vector thne in nonoptimised parameter vector if not found give it a value of 1
#    if len(p2) > 2:  # Forcing maxbas to be predefined
#        maxbas = p2[2]  
#    if len(par) >= 11:  # Putting maxbas as parameter to be optimised
#        maxbas = par[len(par)-1]
#    else:
#        maxbas = 1
#    
#    q_tr = _routing(np.array(q_sim), maxbas)
    
    return q_sim, st


##%% errors
#def nse(q_obs, q_sim):
#    '''
#    ===
#    NSE
#    ===
#    
#    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the 
#    hydrological model
#    
#    Parameters
#    ----------
#    q_rec : array_like [n]
#        Measured discharge [m3/s]
#    q_sim : array_like [n] 
#        Simulated discharge [m3/s]
#    e=error.explained_variance_score(q_obs,q_sim)    
#    Returns
#    -------
#    f : float
#        NSE value
#    '''
##    e=error.explained_variance_score(q_obs,q_sim)
#    a=sum((q_obs-q_sim)**2)
#    b=sum((q_obs-np.average(q_obs))**2)
#    e=1-(a/b)
##    a = np.square(np.subtract(q_rec, q_sim))
##    b = np.square(np.subtract(q_rec, np.nanmean(q_rec))) # to ignore nan values 
##    if a.any < 0.0:
##        return(np.nan)
##    f = 1.0 - (np.nansum(a)/np.nansum(b))
#    return e

"""
def calibrate(flow, avg_prec, temp, et, p2, init_st=None, ll_temp=None,
              x_0=None, x_lb=P_LB, x_ub=P_UB, obj_fun=_rmse, wu=10,
              verbose=False, tol=0.001, minimise=True, fun_nam='RMSE'):
    '''
    =========
    Calibrate
    =========
    Run the calibration of the HBV-96. The calibration is used to estimate the
    optimal set of parameters that minimises the difference between 
    observations and modelled discharge.
    
    Parameters
    ----------
    
    flow : array_like [n]
        Measurements of discharge [m3/s]
    avg_prec : array_like [n]
        Average precipitation [mm/h]
    temp : array_like [n]
        Average temperature [C]
    et : array_like [n]
        Potential Evapotranspiration [mm/h] 
    p2 : array_like [3]
        Problem parameter vector setup as:
        [tfac, area, Lake_area]
    init_st : array_like [5], optional
        Initial model states, [sp, sm, uz, lz, wc]. If unspecified, 
        [0.0, 30.0, 30.0, 30.0, 0.0] mm
    ll_temp : array_like [n], optional
        Long term average temptearature. If unspecified, calculated from temp.
    x_0 : array_like [11], optional
        First guess of the parameter vector. If unspecified, a random value
        will be sampled between the boundaries of the 
    x_lb : array_like [11], optional
        Lower boundary of the parameter vector. If unspecified, a random value
        will be sampled between the boundaries of the 
    x_ub : array_like [11], optional
        First guess of the parameter vector. If unspecified, a random value
        will be sampled between the boundaries of the
    obj_fun : function, optional
        Function that takes 2 parameters, recorded and simulated discharge. If
        unspecified, RMSE is used.
    wu : int, optional
        Warming up period. This accounts for the number of steps that the model
        is run before calculating the performance function.
    verbose : bool, optional
        If True, displays the result of each model evaluation when performing
        the calibration of the hydrological model.
    tol : float, optional
        Determines the tolerance of the solutions in the optimisaiton process.
    minimise : bool, optional
        If True, the optimisation corresponds to the minimisation of the 
        objective function. If False, the optimial of the objective function is
        maximised.
    fun_nam : str, optional
        Name of the objective function used in calibration. If unspecified, is
        'RMSE'
    
    Returns
    -------
    params : array_like [11]
        Optimal parameter set
    
    performance : float
        Optimal value of the objective function
    '''

    def _cal_fun(par):
        q_sim = simulate(avg_prec[:-1], temp, et, par, p2, init_st=None,
                         ll_temp=None, q_0=10.0)[0]
        
        # calculate the performance function
        if minimise:
            perf = obj_fun(flow[wu:], q_sim[wu:])
        else:
            perf = -obj_fun(flow[wu:], q_sim[wu:])

        if verbose:
            print('{0}: {1}'.format(fun_nam, perf))
        return perf

    # Boundaries
    x_b = zip(x_lb, x_ub)

    # initial guess
    if x_0 is None:
        # Randomly generated
        x_0 = np.random.uniform(x_lb, x_ub)

    # Model optimisation
    par_cal = opt.minimize(_cal_fun, x_0, method='L-BFGS-B', bounds=x_b,
                           tol=tol)
    #'L-BFGS-B','Newton-CG', 'nelder-mead', 'trust-ncg'
    params = par_cal.x
    performance = par_cal.fun
    return params, performance
"""