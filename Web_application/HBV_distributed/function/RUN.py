# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
#%links

#%library
import os
#import numpy as np
import gdal
#import matplotlib.pyplot as plt

# functions
import Wrapper
import GISpy as GIS
#import DistParameters as Dp
#import GISCatchment as GC

def RunModel(ConceptualModel, Paths, ParPath, p2, init_st, snow):
    """
    =======================================================================
        RunModel(PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ParPath, p2)
    =======================================================================
    this function runs the conceptual distributed hydrological model
    
    Inputs:
        1-Paths:
            1-PrecPath:
                [String] path to the Folder contains precipitation rasters
            2-Evap_Path:
                [String] path to the Folder contains Evapotranspiration rasters
            3-TempPath:
                [String] path to the Folder contains Temperature rasters
            4-FlowAccPath:
                [String] path to the Flow Accumulation raster of the catchment (it should
                include the raster name and extension)
            5-FlowDPath:
                [String] path to the Flow Direction raster of the catchment (it should
                include the raster name and extension)
        7-ParPath:
            [String] path to the Folder contains parameters rasters of the catchment 
        8-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
    Outputs:
        1- st:
            [4D array] state variables
        2- q_out:
            [1D array] calculated Discharge at the outlet of the catchment
        3- q_uz:
            [3D array] Distributed discharge for each cell
    Example:
        PrecPath = prec_path="meteodata/4000/calib/prec"
        Evap_Path = evap_path="meteodata/4000/calib/evap"
        TempPath = temp_path="meteodata/4000/calib/temp"
        DemPath = "GIS/4000/dem4000.tif"
        FlowAccPath = "GIS/4000/acc4000.tif"
        FlowDPath = "GIS/4000/fd4000.tif"
        ParPath = "meteodata/4000/parameters"
        p2=[1, 227.31]
        st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                          FlowAccPath,FlowDPath,ParPath,p2)
    """
    # input data validation
    assert len(Paths) == 5, "Paths should include 5 folder pathes " +str(len(Paths))+" paths are only provided"
    
    PrecPath=Paths[0]
    Evap_Path=Paths[1]
    TempPath=Paths[2]
#    DemPath=Paths[3]
    FlowAccPath=Paths[3]
    FlowDPath=Paths[4]
    
    # data type
    assert type(PrecPath)== str, "PrecPath input should be string type"
    assert type(Evap_Path)== str, "Evap_Path input should be string type"
    assert type(TempPath)== str, "TempPath input should be string type"
#    assert type(DemPath)== str, "DemPath input should be string type"
    assert type(FlowAccPath)== str, "FlowAccPath input should be string type"
    assert type(FlowDPath)== str, "FlowDPath input should be string type"
    assert type(ParPath)== str, "ParPath input should be string type"
    
    
    # input values
#    dem_ext=DemPath[-4:]
#    assert dem_ext == ".tif", "please add the extension at the end of the DEM raster path input"
    acc_ext=FlowAccPath[-4:]
    assert acc_ext == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
    fd_ext=FlowDPath[-4:]
    assert fd_ext == ".tif", "please add the extension at the end of the Flow Direction path input"
    # check wether the path exists or not
    assert os.path.exists(PrecPath), PrecPath + " you have provided does not exist"
    assert os.path.exists(Evap_Path), Evap_Path+" path you have provided does not exist"
    assert os.path.exists(TempPath), TempPath+" path you have provided does not exist"
#    assert os.path.exists(DemPath), DemPath+ " you have provided does not exist"
    assert os.path.exists(FlowAccPath), FlowAccPath + " you have provided does not exist"
    assert os.path.exists(FlowDPath), FlowDPath+ " you have provided does not exist"
    # check wether the folder has the rasters or not 
    assert len(os.listdir(PrecPath)) > 0, PrecPath+" folder you have provided is empty"
    assert len(os.listdir(Evap_Path)) > 0, Evap_Path+" folder you have provided is empty"
    assert len(os.listdir(TempPath)) > 0, TempPath+" folder you have provided is empty"
    
    # read data
    ### meteorological data
    prec=GIS.ReadRastersFolder(PrecPath)
    evap=GIS.ReadRastersFolder(Evap_Path)
    temp=GIS.ReadRastersFolder(TempPath)
    print("meteorological data are read successfully")
    
    #### GIS data
#    dem= gdal.Open(DemPath) 
    acc=gdal.Open(FlowAccPath)
    fd=gdal.Open(FlowDPath)
    print("GIS data are read successfully")
    
    # parameters
    parameters=GIS.ReadRastersFolder(ParPath)
    print("Parameters are read successfully")
    
    #run the model
    st, q_out, q_uz, q_lz = Wrapper.Dist_model(ConceptualModel, acc, fd, prec, evap,
                                               temp, parameters, p2, snow, init_st)
    
    return st, q_out, q_uz, q_lz