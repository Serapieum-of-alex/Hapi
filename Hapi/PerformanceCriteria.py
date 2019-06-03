import numpy as np
import numbers

"""
====================
WRMSE
====================
Weighted root mean square error
Functions
        1- rmseHF
        2- rmseLF
"""
def RMSE(Qobs,Qsim):
    """
    ===========================================================
        RMSE
    ===========================================================
    
    Root Mean Squared Error. Metric for the estimation of performance of the 
    hydrological model.
    
    Inputs:
    ----------
        1-Qobs :
            [numpy ndarray] Measured discharge [m3/s]
        2-Qsim :
            [numpy ndarray] Simulated discharge [m3/s]
    
    Outputs:
    -------
        1-error : 
            [float] RMSE value
    """
    # convert Qobs & Qsim into arrays
    Qobs=np.array(Qobs)
    Qsim=np.array(Qsim)
    
    rmse = np.sqrt(np.average((np.array(Qobs)-np.array(Qsim))** 2))

    return rmse

def RMSEHF(Qobs,Qsim,WStype,N,alpha):
    """
    ====================
    rmseHF
    ====================
    Weighted Root mean square Error for High flow 
    
    inputs:
    ----------
        1- Qobs: 
            observed flow 
        2- Qsim: 
            simulated flow
        3- WStype:
            Weighting scheme (1,2,3,4)
        4- N:
            power
        5- alpha:
            Upper limit for low flow weight 
    Output:
    ----------
        1- error values
    """
    # input data validation
    # data type
    assert type(WStype)== int, "Weighting scheme should be an integer number between 1 and 4 and you entered "+str(WStype)
    assert isinstance(alpha, numbers.Number), "alpha should be a number and between 0 & 1"
    assert isinstance(N, numbers.Number), "N should be a number and between 0 & 1"
    # Input values
    assert WStype >= 1 and WStype <= 4 , "Weighting scheme should be an integer number between 1 and 4 you have enters "+ str(WStype)
    assert N >= 0 , "Weighting scheme Power should be positive number you have entered "+ str(N)
    assert alpha > 0 and alpha <1, "alpha should be float number and between 0 & 1 you have entered "+ str(alpha)
    
    # convert Qobs & Qsim into arrays
    Qobs=np.array(Qobs)
    Qsim=np.array(Qsim)
    
    
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


def RMSELF(Qobs,Qsim,WStype,N,alpha):
    """
    ====================
    rmseLF
    ====================
    Weighted Root mean square Error for low flow 
    
    inputs:
    ----------
        1- Qobs : observed flow 
        2- Qsim : simulated flow
        3- WStype : Weighting scheme (1,2,3,4)
        4- N: power
        5- alpha : Upper limit for low flow weight 
        
    Output:
    ----------
        1- error values
    """
    # input data validation
    # data type
    assert type(WStype)== int, "Weighting scheme should be an integer number between 1 and 4 and you entered "+str(WStype)
    assert isinstance(alpha, numbers.Number), "alpha should be a number and between 0 & 1"
    assert isinstance(N, numbers.Number), "N should be a number and between 0 & 1"
    # Input values
    assert WStype >= 1 and WStype <= 4 , "Weighting scheme should be an integer number between 1 and 4 you have enters "+ str(WStype)
    assert N >= 0 , "Weighting scheme Power should be positive number you have entered "+ str(N)
    assert alpha > 0 and alpha <1, "alpha should be float number and between 0 & 1 you have entered "+ str(alpha)
    
    # convert Qobs & Qsim into arrays
    Qobs=np.array(Qobs)
    Qsim=np.array(Qsim)
    
    
    Qmax=max(Qobs)  # rational Discharge power N
    l= (Qmax-Qobs)/Qmax
    
    if WStype==1:
         w = l**N
    elif WStype==2: #------------------------------- N is not in the equation
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
    ----------
        1- Qobs : observed flow 
        2- Qsim : simulated flow
    
    Output:
    ----------
        1- error values
    """
    # convert Qobs & Qsim into arrays
    Qobs=np.array(Qobs)
    Qsim=np.array(Qsim)
    
    c= np.corrcoef(Qobs,Qsim)[0][1]
    alpha=np.std(Qsim)/np.std(Qobs)
    beta= np.mean(Qsim)/np.mean(Qobs)
    
    kge=1-np.sqrt(((c-1)**2)+((alpha-1)**2)+((beta-1)**2))
    
    return kge


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
    ----------
        1- Qobs : observed flow 
        2- Qsim : simulated flow
    
    Output:
    ----------
        1- error values
    """
    Qobssum=np.sum(Qobs)
    Qsimsum=np.sum(Qsim)
    wb=100*(1-np.abs(1-(Qsimsum/Qobssum)))
    
    return wb


def NSE(Qobs, Qsim):
    """
    =================================================
        NSE(Qobs, Qsim)
    =================================================
    
    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the 
    hydrological model
    
    Inputs:
    ----------
        1-Qobs :
            [numpy ndarray] Measured discharge [m3/s]
        2-Qsim : 
            [numpy ndarray] Simulated discharge [m3/s]
        
    Outputs
    -------
        1-f :
            [float] NSE value
    
    Examples:
    -------    
        Qobs=np.loadtxt("Qobs.txt")
        Qout=Model(prec,evap,temp)
        error=NSE(Qobs,Qout)
    """
    # convert Qobs & Qsim into arrays
    Qobs=np.array(Qobs)
    Qsim=np.array(Qsim)

    a=sum((Qobs-Qsim)**2)
    b=sum((Qobs-np.average(Qobs))**2)
    e=1-(a/b)

    return e