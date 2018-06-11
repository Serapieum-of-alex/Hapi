 #%% library
import numpy as np

"""
====================
WRMSE
====================
Weighted root mean square error
Functions
        1- rmseHF
        2- rmseLF
"""
def rmse(q_obs,q_sim):
    '''
    ====
    RMSE
    ====
    
    Root Mean Squared Error. Metric for the estimation of performance of the 
    hydrological model.
    
    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n] 
        Simulated discharge [m3/s]
    rmse = np.sqrt(error.mean_squared_error(q_obs,q_sim))    
    Returns
    -------
    f : float
        RMSE value
    '''
#    rmse = np.sqrt(error.mean_squared_error(q_obs,q_sim))
    rmse = np.sqrt(np.average((q_obs-q_sim)** 2))
#    erro = np.square(np.subtract(q_rec,q_sim))
#    if erro.any < 0:
#        return(np.nan)
#    f = np.sqrt(np.nanmean(erro))
    return rmse

def rmseHF(Qobs,Qsim,WStype,N,alpha):
    """
    ====================
    rmseHF
    ====================
    Weighted Root mean square Error for High flow 
    inputs:
        1- Qobs : observed flow 
        2- Qsim : simulated flow
        3- WStype : Weighting scheme (1,2,3,4)
        4- N: power
        5- alpha : Upper limit for low flow weight 
    Output:
        1- error values

    """
    Qmax=max(Qobs)
    h=Qobs/Qmax # rational Discharge 
    
    if WStype==1:
         w = h**N        # rational Discharge power N
    elif WStype==2: #-------------------------------------------------------------N is not in the equation
        w=(h/alpha)**N  
        w[h>alpha] = 1
    elif WStype==3:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha 
        w[h>alpha] = 1
    elif WStype==4:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha 
        w[h>alpha] = 1
    else:                      # sigmoid function
        w=1/(1+np.exp(-10*h+5))
        
    a= (Qobs-Qsim)**2
    b=a*w
    c=sum(b)
    error=np.sqrt(c/len(Qobs))
    
    return error
#______________________________________________________________________________

def rmseLF(Qobs,Qsim,WStype,N,alpha):
    """
    ====================
    rmseLF
    ====================
    Weighted Root mean square Error for low flow 
    inputs:
        1- Qobs : observed flow 
        2- Qsim : simulated flow
        3- WStype : Weighting scheme (1,2,3,4)
        4- N: power
        5- alpha : Upper limit for low flow weight 
    Output:
        1- error values
    """
    Qmax=max(Qobs)  # rational Discharge power N
    l= (Qmax-Qobs)/Qmax
    
    if WStype==1:
         w = l**N
    elif WStype==2: #------------------------------------------------------------ N is not in the equation
#        w=1-l*((0.50 - alpha)**N)
        w=((1/(alpha**2))*(1-l)**2)-((2/alpha)*(1-l))+1
        w[1-l> alpha]=0
    elif WStype==3:   # the same like WStype 2
#        w=1-l*((0.50 - alpha)**N)
        w=((1/(alpha**2))*(1-l)**2)-((2/alpha)*(1-l))+1
        w[1-l> alpha]=0
    elif WStype==4:
    #        w = 1-l*(0.50 - alpha) 
        w= 1 - ((1-l)/alpha)
        w[1-l>alpha] = 0
    else:                     # sigmoid function
#        w=1/(1+np.exp(10*h-5))
        w=1/(1+np.exp(-10*l+5))
        
    a= (Qobs-Qsim)**2
    b=a*w
    c=sum(b)
    error=np.sqrt(c/len(Qobs))
    
    return error
#______________________________________________________________________________
    
def KGE(Qobs,Qsim):
    """
    ====================
    KGE 
    ====================
    (Gupta et al. 2009) have showed the limitation of using a single error
    function to measure the efficiency of calculated flow and showed that
    Nash-Sutcliff efficiency (NSE) or RMSE can be decomposed into three component
    correlation, variability and bias.
    inputs:
        1- Qobs : observed flow 
        2- Qsim : simulated flow
    Output:
        1- error values
    """
    c= np.corrcoef(Qobs,Qsim)[0][1]
    alpha=np.std(Qsim)/np.std(Qobs)
    beta= np.mean(Qsim)/np.mean(Qobs)
    
    kge=1-np.sqrt(((c-1)**2)+((alpha-1)**2)+((beta-1)**2))
    
    return kge
#______________________________________________________________________________
def WB(Qobs,Qsim):
    """
    ====================
    WB
    ====================
    The mean cumulative error measures how much the model succeed to reproduce
    the stream flow volume correctly. This error allows error compensation from
    time step to another and it is not an indication on how accurate is the model
    in the simulated flow. the naive model of Nash-Sutcliffe (simulated flow is
    as accurate as average observed flow) will result in WB error equals to 100 %.
    (Oudin et al. 2006)
    inputs:
        1- Qobs : observed flow 
        2- Qsim : simulated flow
    Output:
        1- error values
    """
    Qobssum=np.sum(Qobs)
    Qsimsum=np.sum(Qsim)
    wb=100*(1-np.abs(1-(Qsimsum/Qobssum)))
    
    return wb

def nse(q_obs, q_sim):
    '''
    ===
    NSE
    ===
    
    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the 
    hydrological model
    
    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n] 
        Simulated discharge [m3/s]
    e=error.explained_variance_score(q_obs,q_sim)    
    Returns
    -------
    f : float
        NSE value
    '''
#    e=error.explained_variance_score(q_obs,q_sim)
    a=sum((q_obs-q_sim)**2)
    b=sum((q_obs-np.average(q_obs))**2)
    e=1-(a/b)
#    a = np.square(np.subtract(q_rec, q_sim))
#    b = np.square(np.subtract(q_rec, np.nanmean(q_rec))) # to ignore nan values 
#    if a.any < 0.0:
#        return(np.nan)
#    f = 1.0 - (np.nansum(a)/np.nansum(b))
    return e