# -*- coding: utf-8 -*-
"""
Created on Wed May 16 03:50:00 2018

@author: Mostafa
"""
#%links

#%library
import os
import datetime as dt
import numpy as np
import shutil
# functions
import GISpy as GIS


def PrepareInputs(Raster,InputFolder,FolderName):
    """
    ================================================================
        PrepareInputs(Raster,InputFolder,FolderName)
    ================================================================
    this function prepare downloaded raster data to have the same align and 
    nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
    and return a folder with the output rasters with a name "New_Rasters"
    
    Inputs:
        1-Raster:
            [String] path to the spatial information source raster to get the spatial information 
            (coordinate system, no of rows & columns) A_path should include the name of the raster 
            and the extension like "data/dem.tif"
        2-InputFolder:
            [String] path of the folder of the rasters you want to adjust their 
            no of rows, columns and resolution (alignment) like raster A 
            the folder should not have any other files except the rasters
        3-FolderName:
            [String] name to create a folder to store resulted rasters
    Example:
        Ex1:
            dem_path="01GIS/inputs/4000/acc4000.tif"
            prec_in_path="02Precipitation/CHIRPS/Daily/"
            Inputs.PrepareInputs(dem_path,prec_in_path,"prec")
        Ex2:
            dem_path="01GIS/inputs/4000/acc4000.tif"
            outputpath="00inputs/meteodata/4000/"
            evap_in_path="03Weather_Data/evap/"
            Inputs.PrepareInputs(dem_path,evap_in_path,outputpath+"evap")
    """
    # input data validation
    # data type
    assert type(FolderName)== str, "FolderName input should be string type"
    # create a new folder for new created alligned rasters in temp
    # check if you can create the folder 
    try:
        os.makedirs(os.path.join(os.environ['TEMP'],"AllignedRasters"))
    except WindowsError : 
        # if not able to create the folder delete the folder with the same name and create one empty
        shutil.rmtree(os.path.join(os.environ['TEMP']+"/AllignedRasters"))
        os.makedirs(os.path.join(os.environ['TEMP'],"AllignedRasters"))
        
    temp=os.environ['TEMP']+"/AllignedRasters/"
    
    # match alignment 
    GIS.MatchDataAlignment(Raster,InputFolder,temp)
    # create new folder in the current directory for alligned and nodatavalue matched cells
    try:
        os.makedirs(os.path.join(os.getcwd(),FolderName))
    except WindowsError:
        print("please function is trying to create a folder with a name New_Rasters to complete the process if there is a folder with the same name please rename it to other name")    
    # match nodata value 
    GIS.MatchDataNoValuecells(Raster,temp,FolderName+"/")
    # delete the processing folder from temp
    shutil.rmtree(temp)

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