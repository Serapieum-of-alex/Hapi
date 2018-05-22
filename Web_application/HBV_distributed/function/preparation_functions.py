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


#% 

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













