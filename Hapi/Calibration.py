# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:53:51 2018

@author: Mostafa
"""
#%links

#%library
import os
import numpy as np
import gdal

from pyOpt import Optimization, ALHSO,Optimizer


# functions
import GISpy as GIS
import DistParameters as Dp
import PerformanceCriteria as PC
import Wrapper


def RunCalibration(Paths,p2,Q_obs,UB,LB,lumpedParNo,lumpedParPos,
                   objective_function,printError=None,*args):
    """
    =======================================================================
        RunCalibration(Paths, p2, Q_obs, UB, LB, lumpedParNo, lumpedParPos, objective_function, printError=None, *args):
    =======================================================================
    this function runs the conceptual distributed hydrological model
    
    Inputs:
    ----------
        1-Paths:
            1-PrecPath:
                [String] path to the Folder contains precipitation rasters
            2-Evap_Path:
                [String] path to the Folder contains Evapotranspiration rasters
            3-TempPath:
                [String] path to the Folder contains Temperature rasters
            4-DemPath:
                [String] path to the DEM raster of the catchment (it should
                include the raster name and extension)
            5-FlowAccPath:
                [String] path to the Flow Accumulation raster of the catchment (it should
                include the raster name and extension)
            6-FlowDPath:
                [String] path to the Flow Direction raster of the catchment (it should
                include the raster name and extension)
        2-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
        3-Q_obs:
            [Numeric] Observed values of discharge 
        4-UB:
            [Numeric] upper bound of the values of the parameters
        5-LB:
            [Numeric] Lower bound of the values of the parameters
        6-lumpedParNo:
            [int] nomber of lumped parameters, you have to enter the value of 
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        7-lumpedParPos:
            [List] list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length 
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        8-objective_function:
            [function] objective function to calculate the performance of the model
            and to be used in the calibration
        9-*args:
            other arguments needed on the objective function
            
    Outputs:
    ----------
        1- st:
            [4D array] state variables
        2- q_out:
            [1D array] calculated Discharge at the outlet of the catchment
        3- q_uz:
            [3D array] Distributed discharge for each cell
    
    Example:
    ----------
        PrecPath = prec_path="meteodata/4000/calib/prec"
        Evap_Path = evap_path="meteodata/4000/calib/evap"
        TempPath = temp_path="meteodata/4000/calib/temp"
        DemPath = "GIS/4000/dem4000.tif"
        FlowAccPath = "GIS/4000/acc4000.tif"
        FlowDPath = "GIS/4000/fd4000.tif"
        ParPath = "meteodata/4000/"+"parameters.txt"
        p2=[1, 227.31]
        st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                          FlowAccPath,FlowDPath,ParPath,p2)
    """
    PrecPath=Paths[0]
    Evap_Path=Paths[0]
    TempPath=Paths[0]
    DemPath=Paths[0]
    FlowAccPath=Paths[0]
    FlowDPath=Paths[0]
    ### inputs validation
    
    # data type
    assert type(PrecPath)== str, "PrecPath input should be string type"
    assert type(Evap_Path)== str, "Evap_Path input should be string type"
    assert type(TempPath)== str, "TempPath input should be string type"
    assert type(DemPath)== str, "DemPath input should be string type"
    assert type(FlowAccPath)== str, "FlowAccPath input should be string type"
    assert type(FlowDPath)== str, "FlowDPath input should be string type"
    
    # input values
    dem_ext=DemPath[-4:]
    assert dem_ext == ".tif", "please add the extension at the end of the DEM raster path input"
    acc_ext=FlowAccPath[-4:]
    assert acc_ext == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
    fd_ext=FlowDPath[-4:]
    assert fd_ext == ".tif", "please add the extension at the end of the Flow Direction path input"
    
    # check wether the path exists or not
    assert os.path.exists(PrecPath), PrecPath + " you have provided does not exist"
    assert os.path.exists(Evap_Path), Evap_Path+" path you have provided does not exist"
    assert os.path.exists(TempPath), TempPath+" path you have provided does not exist"
    assert os.path.exists(DemPath), DemPath+ " you have provided does not exist"
    assert os.path.exists(FlowAccPath), FlowAccPath + " you have provided does not exist"
    assert os.path.exists(FlowDPath), FlowDPath+ " you have provided does not exist"
    
    # check wether the folder has the rasters or not 
    assert len(os.listdir(PrecPath)) > 0, PrecPath+" folder you have provided is empty"
    assert len(os.listdir(Evap_Path)) > 0, Evap_Path+" folder you have provided is empty"
    assert len(os.listdir(TempPath)) > 0, TempPath+" folder you have provided is empty"
    
    # check objective_function
    assert callable(objective_function) , "second argument should be a function"
    
    if args== None :
        args=[]
    
    
    # read data
    ### meteorological data
    prec=GIS.ReadRastersFolder(PrecPath)
    evap=GIS.ReadRastersFolder(Evap_Path)
    temp=GIS.ReadRastersFolder(TempPath)
    print("meteorological data are read successfully")
    #### GIS data
    dem= gdal.Open(DemPath) 
    acc=gdal.Open(FlowAccPath)
    fd=gdal.Open(FlowDPath)
    print("GIS data are read successfully")
    init_st=[0,5,5,5,0]
    
    ### optimization
    # generate random parameters for the first run
    par=np.random.uniform(LB, UB)
    print('Calibration starts')
    ### calculate the objective function
    def opt_fun(par):
        try:
            # parameters
            klb=float(par[-2])
            kub=float(par[-1])
            par=par[:-2]
            # distribute the parameters to a 2d array
            par2d=Dp.par3d(par,dem,12,lumpedParNo,lumpedParPos,kub,klb)    
            #run the model
            st, q_out, q_uz=Wrapper.Dist_model(dem,acc,fd,prec,evap,temp,par2d,p2,init_st)
            # calculate performance of the model
            error=objective_function(Q_obs,q_out,*args)
            # print error
            if printError != None:
                print(error)
                print(par)
            
            fail = 0
        except:
            error = np.nan
            fail = 1
            
        return error, [], fail 
    
    ### define the optimization components
    opt_prob = Optimization('HBV Calibration', opt_fun)
    for i in range(len(LB)):
        opt_prob.addVar('x{0}'.format(i), type='c', lower=LB[i], upper=UB[i])
    
    print(opt_prob)
    
    opt_engine = ALHSO(etol=0.0001,atol=0.0001,rtol=0.0001, stopiters=10,
                       hmcr=0.5,par=0.5,filename='mostafa.out')
    
    Optimizer.__init__(opt_engine,def_options={
                    'hms':[int,9],					# Memory Size [1,50]
                		'hmcr':[float,0.95],			# Probability rate of choosing from memory [0.7,0.99]
                		'par':[float,0.99],				# Pitch adjustment rate [0.1,0.99]
                		'dbw':[int,2000],				# Variable Bandwidth Quantization
                		'maxoutiter':[int,2e3],			# Maximum Number of Outer Loop Iterations (Major Iterations)
                		'maxinniter':[int,2e2],			# Maximum Number of Inner Loop Iterations (Minor Iterations)
                		'stopcriteria':[int,1],			# Stopping Criteria Flag
                		'stopiters':[int,20],			# Consecutively Number of Outer Iterations for which the Stopping Criteria must be Satisfied
                		'etol':[float,0.0001],			# Absolute Tolerance for Equality constraints
                		'itol':[float,0.0001],			# Absolute Tolerance for Inequality constraints 
                		'atol':[float,0.0001],			# Absolute Tolerance for Objective Function 1e-6
                		'rtol':[float,0.0001],			# Relative Tolerance for Objective Function
                		'prtoutiter':[int,0],			# Number of Iterations Before Print Outer Loop Information
                		'prtinniter':[int,0],			# Number of Iterations Before Print Inner Loop Information
                		'xinit':[int,0],				# Initial Position Flag (0 - no position, 1 - position given)
                		'rinit':[float,1.0],			# Initial Penalty Factor
                		'fileout':[int,1],				# Flag to Turn On Output to filename
                		'filename':[str,'parameters.txt'],	# We could probably remove fileout flag if filename or fileinstance is given
                		'seed':[float,0.5],				# Random Number Seed (0 - Auto-Seed based on time clock)
                		'scaling':[int,1],				# Design Variables Scaling Flag (0 - no scaling, 1 - scaling between [-1,1]) 
                		})
    
    res = opt_engine(opt_prob)
    
    
    return res
