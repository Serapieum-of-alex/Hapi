# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:42:04 2018

@author: Mostafa
"""
#%links

#%library
import numpy as np


# functions


#% 

def muskingum(inflow,Qinitial,k,x,dt):
# =============================================================================
#     inputs:
#            1- inflow
#            2- Qinitial initial value for outflow
#            3- k travelling time (hours)
#            4- x surface nonlinearity coefficient (0,0.5)
#            5- dt delta t 
# =============================================================================
    
    c1=(dt-2*k*x)/(2*k*(1-x)+dt)
    c2=(dt+2*k*x)/(2*k*(1-x)+dt)
    c3=(2*k*(1-x)-dt)/(2*k*(1-x)+dt)
    
#    if c1+c2+c3!=1:
#        raise("sim of c1,c2 & c3 is not 1")
    
    outflow=np.ones_like(inflow)*np.nan    
    outflow[0]=Qinitial
    
    for i in range(1,len(inflow)):
        outflow[i]=c1*inflow[i]+c2*inflow[i-1]+c3*outflow[i-1]
    
#    outflow[outflow<0]=0
    outflow=np.round(outflow,4)
    return outflow






