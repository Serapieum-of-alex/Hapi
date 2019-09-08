# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 23:55:20 2018

@author: Mostafa
"""


#%library
import pickle
import datetime
#import numpy as np

# functions
def save_obj(obj, saved_name ):
    """
    ===============================================================
        save_obj(obj, saved_name )
    ===============================================================
    this function is used to save any python object to your hard desk
    
    Inputs:
    ----------
        1-obj:
            
        2-saved_name:
            ['String'] name of the object 
    Outputs:    
    ----------
        the object will be saved to the given path/current working directory
        with the given name
    Example:
        data={"key1":[1,2,3,5],"key2":[6,2,9,7]}
        save_obj(data,path+'/flow_acc_table')
    """
    with open( saved_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(saved_name):
    """
    ===============================================================
        load_obj(saved_name)
    ===============================================================
    this function is used to save any python object to your hard desk
    
    Inputs:
    ----------
        1-saved_name:
            ['String'] name of the object
    Outputs:    
    ----------
        the object will be loaded
    Example:
        load_obj(path+'/flow_acc_table')
    """
    with open( saved_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def DateFormatedSQL(x):
    """
    ===========================================================
        DateFormatedSQL(x)
    ===========================================================
    this function converts the the date read from a list to a datetime format
    
    input:
    ----------
        [List] list of tuples of string date read from database
    
    output:
    ----------
        [List] list of dates as a datetime format  YYYY-MM-DD HH:MM:SS
    
    """
    x=[i[0] for i in x]
    
    x1=[]
    for i in x:
        if len(i)==19:
            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[14:16]),int(i[17:18]) ))
#        elif len(i)==13:
#            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(0),int(0) ))
#        else:
#            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(0),int(0),int(0) ))
#    del i,x
    return x1

def DateFormated(x):
    """
    ===========================================================
        dateformated(x)
    ===========================================================
    this function converts the the date read from a list to a datetime format
    
    input:
    ----------
        [List] list of dates as string
    
    output:
    ----------
        [List] list of dates as a datetime format YYYY-MM-DD HH:MM:SS
    """
    
    x1=[]
    for i in x:
        if len(i)==19:
            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(i[14:16]),int(i[17:18]) ))
#        elif len(i)==13:
#            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(0),int(0) ))
#        else:
#            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(0),int(0),int(0) ))
#    del i,x
    return x1