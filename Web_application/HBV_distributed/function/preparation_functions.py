# -*- coding: utf-8 -*-
"""
Created on Wed May 16 03:50:00 2018

@author: Mostafa
"""
#%links

#%library
import datetime as dt
import numpy as np
# functions




def changetext2time(string):
    """
    # =============================================================================
    #     changetext2time(string)
    # =============================================================================
    this functions changes the date from a string to a date time format
    """
    time=dt.datetime(int(string[:4]),int(string[5:7]),int(string[8:10]),
                  int(string[11:13]),int(string[14:16]),int(string[17:]))
    return time

def rescale(OldValue,OldMin,OldMax,NewMin,NewMax):
    """
    # =============================================================================
    #  rescale(OldValue,OldMin,OldMax,NewMin,NewMax)
    # =============================================================================
    this function rescale a value between two boundaries to a new value bewteen two 
    other boundaries
    inputs:
        1-OldValue:
            [float] value need to transformed
        2-OldMin:
            [float] min old value
        3-OldMax:
            [float] max old value
        4-NewMin:
            [float] min new value
        5-NewMax:
            [float] max new value
    output:
        1-NewValue:
            [float] transformed new value
        
    """
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    
    return NewValue


def mycolor(x,min_old,max_old,min_new, max_new):
    """
    # =============================================================================
    #  mycolor(x,min_old,max_old,min_new, max_new)
    # =============================================================================
    this function transform the value between two normal values to a logarithmic scale
    between logarithmic value of both boundaries 
    inputs:
        1-x:
            [float] new value needed to be transformed to a logarithmic scale
        2-min_old:
            [float] min old value in normal scale
        3-max_old:
            [float] max old value in normal scale
        4-min_new:
            [float] min new value in normal scale
        5-max_new:
            [float] max_new max new value
    output:
        1-Y:
            [int] integer number between new max_new and min_new boundaries
    """
    
    # get the boundaries of the logarithmic scale
    if min_old== 0.0:
        min_old_log=-7
    else:
        min_old_log=np.log(min_old)
        
    max_old_log=np.log(max_old)    
    
    if x==0:
        x_log=-7
    else:
        x_log=np.log(x)
    
    y=int(np.round(rescale(x_log,min_old_log,max_old_log,min_new,max_new)))
    
    return y