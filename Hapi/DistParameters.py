# -*- coding: utf-8 -*-
"""
DistParameters contains functions that is responible for distributing parameters
spatially (totally distributed, totally distriby-uted with some parameters lumped,
all parameters are lumped, hydrologic response units) and also save generated parameters 
into rasters 

@author: Mostafa
"""

# library
import numbers
import numpy as np
import os
import gdal

import GISpy as GIS
import GISCatchment as GC


# functions
def calculateK(x,position,UB,LB):
    """
    ===================================================
        calculateK(x,position,UB,LB):
    ===================================================
        
    this function takes value of x parameter and generate 100 random value of k parameters between
    upper & lower constraint then the output will be the value coresponding to the giving position
    
    Inputs:
    ----------
        1- x weighting coefficient to determine the linearity of the water surface
            (one of the parameters of muskingum routing method)
        2- position 
            random position between upper and lower bounds of the k parameter
        3-UB 
            upper bound for k parameter
        3-LB 
            Lower bound for k parameter
    """
    
    constraint1=0.5*1/(1-x) # k has to be smaller than this constraint
    constraint2=0.5*1/x   # k has to be greater than this constraint
    
    if constraint2 >= UB : #if constraint is higher than UB take UB
        constraint2 =UB
        
    if constraint1 <= LB : #if constraint is lower than LB take UB
        constraint1 =LB
    
    generatedK=np.linspace(constraint1,constraint2,101)
    k=generatedK[int(round(position,0))]
    return k


def par2d_lumpedK1_lake(par_g,raster,no_parameters,no_parameters_lake,kub,klb):
    """
    ===========================================================
      par2d_lumpedK1_lake(par_g,raster,no_parameters,no_par_lake,kub,klb)
    ===========================================================
    this function takes a list of parameters and distribute them horizontally on number of cells
    given by a raster 
    
    Inputs :
    ----------
        1- par_g
            list of parameters
        2- raster
            raster of the catchment (DEM)
        3- no_parameters
            no of parameters of the cell
        4- no_parameters_lake
            no of lake parameters
        5- kub
            upper bound of K value (traveling time in muskingum routing method)
        6- klb
            Lower bound of K value (traveling time in muskingum routing method)
    
    Output:
    ----------
        1- par_2d: 3D array of the parameters distributed horizontally on the cells
        2- lake_par: list of the lake parameters.
        
    Example:
    ----------
        a list of 155 value,all parameters are distributed except lower zone coefficient
        (is written at the end of the list) each cell(14 cells) has 11 parameter plus lower zone
        (12 parameters) function will take each 11 parameter and assing them to a specific cell
        then assign the last value (lower zone parameter) to all cells
        14*11=154 + 1 = 155
    """
    # get the shape of the raster
    shape_base_dem = raster.ReadAsArray().shape
    # read the raster    
    f=raster.ReadAsArray()
    # get the no_value of in the raster    
    no_val = np.float32(raster.GetRasterBand(1).GetNoDataValue())
    # count the number of non-empty cells 
    no_elem = np.sum(np.sum([[1 for elem in mask_i if elem != no_val] for mask_i in f]))
    
    # store the indeces of the non-empty cells
    celli=[]#np.ones((no_elem,2))
    cellj=[]
    for i in range(shape_base_dem[0]): # rows
        for j in range(shape_base_dem[1]): # columns
            if f[i,j]!= no_val:
                celli.append(i)
                cellj.append(j)
    
    # create an empty 3D array [[raster dimension], no_parameters]
    par_2d=np.zeros([shape_base_dem[0], shape_base_dem[1], no_parameters])*np.nan
    
    # parameters in array
    # remove a place for the lumped parameter (k1) lower zone coefficient    
    no_parameters=no_parameters-1
    
    # create a 2d array [no_parameters, no_cells]
    par_arr=np.ones((no_parameters,no_elem))
    
    # take the parameters from the generated parameters or the 1D list and 
    # assign them to each cell
    for i in range(no_elem):
        par_arr[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]
    
    # create a list with the value of the lumped parameter(k1)
    # (stored at the end of the list of the parameters)
    pk1=np.ones((1,no_elem))*par_g[(np.shape(par_arr)[0]*np.shape(par_arr)[1])]
    
    # put the list of parameter k1 at the 6 row
    par_arr=np.vstack([par_arr[:6,:],pk1,par_arr[6:,:]])
    
    # assign the parameters from the array (no_parameters, no_cells) to 
    # the spatially corrected location in par2d
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],:]=par_arr[:,i]
    
    # calculate the value of k(travelling time in muskingum based on value of 
    # x and the position and upper, lower bound of k value 
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],-2]= calculateK(par_2d[celli[i],cellj[i],-1],par_2d[celli[i],cellj[i],-2],kub,klb)
    
    # lake parameters
    lake_par=par_g[len(par_g)-no_parameters_lake:]
    lake_par[-2]=calculateK(lake_par[-1],lake_par[-2],kub,klb)
    
    return par_2d,lake_par


def par3dLumped(par_g,raster,no_parameters,kub=1,klb=0.5):
    """
    ===========================================================
      par3dLumped(par_g,raster, no_parameters, kub, klb)
    ===========================================================
    this function takes a list of parameters [saved as one column or generated
    as 1D list from optimization algorithm] and distribute them horizontally on
    number of cells given by a raster
    
    Inputs :
    ----------
        1- par_g:
            [list] list of parameters
        2- raster:
            [gdal.dataset] raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        3- no_parameters
            [int] no of parameters of the cell according to the rainfall runoff model
        4- kub:
            [float] upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour 
        5- klb:
            [float] Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)
    
    Output:
    ----------
        1- par_3d: 3D array of the parameters distributed horizontally on the cells
        
    Example:
    ----------
        EX1:Lumped parameters
            raster=gdal.Open("dem.tif")
            [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]    
            
            raster=gdal.Open(path+"soil_classes.tif")
            no_parameters=12
            par_g=np.random.random(no_parameters) #no_elem*(no_parameters-no_lumped_par)
            
            tot_dist_par=DP.par3dLumped(par_g,raster,no_parameters,lumped_par_pos,kub=1,klb=0.5)
    """
    # input data validation
    # data type
    assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
    assert type(no_parameters)==int, " no_parameters should be integer number"
    assert isinstance(kub,numbers.Number) , " kub should be a number"
    assert isinstance(klb,numbers.Number) , " klb should be a number"

    # read the raster    
    raster_A=raster.ReadAsArray()
    # get the shape of the raster
    rows=raster.RasterYSize
    cols=raster.RasterXSize
    # get the no_value of in the raster    
    noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())
    
    # count the number of non-empty cells 
    no_elem = np.size(raster_A[:,:])-np.count_nonzero((raster_A[raster_A==noval])) 
    
    # store the indeces of the non-empty cells
    celli=[]#np.ones((no_elem,2))
    cellj=[]
    for i in range(rows): # rows
        for j in range(cols): # columns
            if raster_A[i,j]!= noval:
                celli.append(i)
                cellj.append(j)
    
    # create an empty 3D array [[raster dimension], no_parameters]
    par_2d=np.zeros([rows, cols, no_parameters])*np.nan
    
    # parameters in array
    # create a 2d array [no_parameters, no_cells]            
    par_arr=np.ones((no_parameters,no_elem))
    
    # take the parameters from the generated parameters or the 1D list and 
    # assign them to each cell
    for i in range(no_elem):
        par_arr[:,i] = par_g #par_g[i*no_parameters:(i*no_parameters)+no_parameters]
    
    # assign the parameters from the array (no_parameters, no_cells) to 
    # the spatially corrected location in par2d
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],:]=par_arr[:,i]
    
    # calculate the value of k(travelling time in muskingum based on value of 
    # x and the position and upper, lower bound of k value 
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],-2]= calculateK(par_2d[celli[i],cellj[i],-1],par_2d[celli[i],cellj[i],-2],kub,klb)
    
    return par_2d


def par3d(par_g,raster,no_parameters,no_lumped_par=0,lumped_par_pos=[],
                   kub=1,klb=0.5):
    """
    ===========================================================
      par3d(par_g,raster, no_parameters, no_lumped_par, lumped_par_pos, kub, klb)
    ===========================================================
    this function takes a list of parameters [saved as one column or generated
    as 1D list from optimization algorithm] and distribute them horizontally on
    number of cells given by a raster
    
    Inputs :
    ----------
        1- par_g:
            [list] list of parameters
        2- raster:
            [gdal.dataset] raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        3- no_parameters
            [int] no of parameters of the cell according to the rainfall runoff model
        4-no_lumped_par:
            [int] nomber of lumped parameters, you have to enter the value of 
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        5-lumped_par_pos:
            [List] list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length 
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        6- kub:
            [float] upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour 
        7- klb:
            [float] Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)
    
    Output:
    ----------
        1- par_3d: 3D array of the parameters distributed horizontally on the cells
        
    Example:
    ----------
        EX1:totally distributed parameters
            raster=gdal.Open("dem.tif")
            [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]    
            no_lumped_par=0
            lumped_par_pos=[]
            par_g=np.random.random(no_elem*(no_parameters-no_lumped_par))
            
            tot_dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
            
        EX2: One Lumped Parameter [K1]
            raster=gdal.Open("dem.tif")
            given values of parameters are of this order
            [fc, beta, etf, lp, c_flux, k, alpha, perc, pcorr, Kmuskingum, Xmuskingum,k1] 
            K1 is lumped so its value is inserted at the end and its order should 
            be after K
            
            no_lumped_par=1
            lumped_par_pos=[6]
            par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
            # insert the value of k1 at the end 
            par_g=np.append(par_g,0.005)
            
            dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
            
        EX3:Two Lumped Parameter [K1, Perc]
            raster=gdal.Open("dem.tif")
            no_lumped_par=2
            lumped_par_pos=[6,8]
            par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
            par_g=np.append(par_g,0.005)
            par_g=np.append(par_g,0.006)
            
            dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
    """
    # input data validation
    # data type
    assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
    assert type(no_parameters)==int, " no_parameters should be integer number"
    assert isinstance(kub,numbers.Number) , " kub should be a number"
    assert isinstance(klb,numbers.Number) , " klb should be a number"
    assert type(no_lumped_par)== int, "no of lumped parameters should be integer"
    
    if no_lumped_par>=1:
        if type(lumped_par_pos)==list:
            assert no_lumped_par==len(lumped_par_pos), "you have to entered"+str(no_lumped_par)+"no of lumped parameters but only"+str(len(lumped_par_pos))+" position "
        else: # if not int or list
            assert 1==5 ,"you have one or more lumped parameters so the position has to be entered as a list"    

    # read the raster    
    raster_A=raster.ReadAsArray()
    # get the shape of the raster
    rows=raster.RasterYSize
    cols=raster.RasterXSize
    # get the no_value of in the raster    
    noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())
    
    # count the number of non-empty cells 
    no_elem = np.size(raster_A[:,:])-np.count_nonzero((raster_A[raster_A==noval])) 
    
    # input values
    if no_lumped_par > 0:
        assert len(par_g)==(no_elem*(no_parameters-no_lumped_par))+no_lumped_par,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
    else:
        # if there is no lumped parameters
        assert len(par_g)==no_elem*no_parameters,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)
    
    # store the indeces of the non-empty cells
    celli=[]#np.ones((no_elem,2))
    cellj=[]
    for i in range(rows): # rows
        for j in range(cols): # columns
            if raster_A[i,j]!= noval:
                celli.append(i)
                cellj.append(j)
    
    # create an empty 3D array [[raster dimension], no_parameters]
    par_2d=np.zeros([rows, cols, no_parameters])*np.nan
    
    # parameters in array
    # remove a place for the lumped parameter (k1) lower zone coefficient    
    no_parameters=no_parameters-no_lumped_par
    
    # create a 2d array [no_parameters, no_cells]            
    par_arr=np.ones((no_parameters,no_elem))
    
    # take the parameters from the generated parameters or the 1D list and 
    # assign them to each cell
    for i in range(no_elem):
        par_arr[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]
    
    ### lumped parameters
    if no_lumped_par > 0:
        for i in range(no_lumped_par):
            # create a list with the value of the lumped parameter(k1)
            # (stored at the end of the list of the parameters)
            pk1=np.ones((1,no_elem))*par_g[(no_parameters*np.shape(par_arr)[1])+i]
            # put the list of parameter k1 at the 6 row    
            par_arr=np.vstack([par_arr[:lumped_par_pos[i],:],pk1,par_arr[lumped_par_pos[i]:,:]])
    
    # assign the parameters from the array (no_parameters, no_cells) to 
    # the spatially corrected location in par2d
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],:]=par_arr[:,i]
    
    # calculate the value of k(travelling time in muskingum based on value of 
    # x and the position and upper, lower bound of k value 
    for i in range(no_elem):
        par_2d[celli[i],cellj[i],-2]= calculateK(par_2d[celli[i],cellj[i],-1],par_2d[celli[i],cellj[i],-2],kub,klb)
    
    return par_2d


def HRU(par_g,raster,no_parameters,no_lumped_par=0,lumped_par_pos=[],
                   kub=1,klb=0.5):
    """
    ===========================================================
       HRU(par_g, raster, no_parameters, no_lumped_par=0, lumped_par_pos=[], kub=1, klb=0.5)
    ===========================================================
    this function takes a list of parameters [saved as one column or generated
    as 1D list from optimization algorithm] and distribute them horizontally on
    number of cells given by a raster
    the input raster should be classified raster (by numbers) into class to be used
    to define the HRUs
    
    Inputs :
    ----------
        1- par_g:
            [list] list of parameters
        2- raster:
            [gdal.dataset] classification raster to get the spatial information
            of the catchment and the to define each cell belongs to which HRU
        3- no_parameters
            [int] no of parameters of the cell according to the rainfall runoff model
        4-no_lumped_par:
            [int] nomber of lumped parameters, you have to enter the value of 
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        5-lumped_par_pos:
            [List] list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length 
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        6- kub:
            [float] upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour 
        7- klb:
            [float] Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)
    
    Output:
    ----------
        1- par_3d: 3D array of the parameters distributed horizontally on the cells
        
    Example:
    ----------
        EX1:HRU without lumped parameters
            [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            raster = gdal.Open("soil_types.tif")
            no_lumped_par=0
            lumped_par_pos=[]
            par_g=np.random.random(no_elem*(no_parameters-no_lumped_par))
            
            par_hru=HRU(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
                
            
        EX2: HRU with one lumped parameters
            given values of parameters are of this order
            [fc, beta, etf, lp, c_flux, k, alpha, perc, pcorr, Kmuskingum, Xmuskingum,k1] 
            K1 is lumped so its value is inserted at the end and its order should 
            be after K 
            
            raster = gdal.Open("soil_types.tif")
            no_lumped_par=1
            lumped_par_pos=[6]
            par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
            # insert the value of k1 at the end 
            par_g=np.append(par_g,0.005)
            
            par_hru=HRU(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
    """
    # input data validation
    # data type
    assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
    assert type(no_parameters)==int, " no_parameters should be integer number"
    assert isinstance(kub,numbers.Number) , " kub should be a number"
    assert isinstance(klb,numbers.Number) , " klb should be a number"
    assert type(no_lumped_par)== int, "no of lumped parameters should be integer"
    
    if no_lumped_par>=1:
        if type(lumped_par_pos)==list:
            assert no_lumped_par==len(lumped_par_pos), "you have to entered"+str(no_lumped_par)+"no of lumped parameters but only"+str(len(lumped_par_pos))+" position "
        else: # if not int or list
            assert 1==5 ,"you have one lumped parameters so the position has to be entered as a list"
    
    
    # read the raster    
    raster_A=raster.ReadAsArray()
    # get the shape of the raster
    rows=raster.RasterYSize
    cols=raster.RasterXSize
    # get the no_value of in the raster    
    noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())

    # count the number of non-empty cells 
    values=list(set([int(raster_A[i,j]) for i in range(rows) for j in range(cols) if raster_A[i,j] != noval]))
    no_elem=len(values)
    
    # input values
    if no_lumped_par > 0:
        assert len(par_g)==(no_elem*(no_parameters-no_lumped_par))+no_lumped_par,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
    else:
        # if there is no lumped parameters
        assert len(par_g)==no_elem*no_parameters,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)
    
    # store the indeces of the non-empty cells
    celli=[]#np.ones((no_elem,2))
    cellj=[]
    for i in range(rows): # rows
        for j in range(cols): # columns
            if raster_A[i,j]!= noval:
                celli.append(i)
                cellj.append(j)
    
    # create an empty 3D array [[raster dimension], no_parameters]
    par_2d=np.zeros([rows, cols, no_parameters])*np.nan
    
    # parameters in array
    # remove a place for the lumped parameter (k1) lower zone coefficient    
    no_parameters=no_parameters-no_lumped_par
    
    # create a 2d array [no_parameters, no_cells]            
    par_arr=np.ones((no_parameters,no_elem))
    
    # take the parameters from the generated parameters or the 1D list and 
    # assign them to each cell
    for i in range(no_elem):
        par_arr[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]
    
    ### lumped parameters
    if no_lumped_par > 0:
        for i in range(no_lumped_par):
            # create a list with the value of the lumped parameter(k1)
            # (stored at the end of the list of the parameters)
            pk1=np.ones((1,no_elem))*par_g[(no_parameters*np.shape(par_arr)[1])+i]
            # put the list of parameter k1 at the 6 row    
            par_arr=np.vstack([par_arr[:lumped_par_pos[i],:],pk1,par_arr[lumped_par_pos[i]:,:]])
    
    # calculate the value of k(travelling time in muskingum based on value of 
    # x and the position and upper, lower bound of k value 
    for i in range(no_elem):
        par_arr[-2,i]= calculateK(par_arr[-1,i],par_arr[-2,i],kub,klb)
    
    # assign the parameters from the array (no_parameters, no_cells) to 
    # the spatially corrected location in par2d each soil type will have the same
    # generated parameters 
    for i in range(no_elem):
        par_2d[raster_A==values[i]]=par_arr[:,i]
        
    return par_2d

def HRU_HAND(DEM,FD,FPL,River):
    """
    =============================================================
        HRU_HAND(DEM,FD,FPL,River)
    =============================================================
    this function calculates inputs for the HAND (height above nearest drainage)
    method for land use classification 
    
    Inputs:
    ----------
        1- DEM:
            
        2-FD:
            
        3-FPL:
            
        4-River:
            
        
    
    Outputs:
    ----------
        1-HAND:
            [numpy ndarray] Height above nearest drainage
            
        2-DTND:
            [numpy ndarray] Distance to nearest drainage
        
    """
    
    # Use DEM raster information to run all loops
    dem_A=DEM.ReadAsArray()
    no_val=np.float32(DEM.GetRasterBand(1).GetNoDataValue())
    rows=DEM.RasterYSize
    cols=DEM.RasterXSize
    
    # get the indices of the flow direction path
    fd_index=GC.FlowDirectIndex(FD)
    
    # read the river location raster
    river_A=River.ReadAsArray()
    
    # read the flow path length raster
    fpl_A=FPL.ReadAsArray()
    
    # trace the flow direction to the nearest river reach and store the location 
    # of that nearst reach
    nearest_network=np.ones((rows,cols,2))*np.nan
    try:
        for i in range(rows):
            for j in range(cols):
                if dem_A[i,j] != no_val:
                    f=river_A[i,j]
                    old_row=i
                    old_cols=j
                    
                    while f != 1:
                        # did not reached to the river yet then go to the next down stream cell
                        # get the down stream cell (furure position)
                        new_row=int(fd_index[old_row,old_cols,0])
                        new_cols=int(fd_index[old_row,old_cols,1])
                        # print(str(new_row)+","+str(new_cols))
                        # go to the downstream cell
                        f=river_A[new_row,new_cols]
                        # down stream cell becomes the current position (old position)
                        old_row=new_row
                        old_cols=new_cols
                        # at this moment old and new stored position are the same (current position)
                    # store the position in the array
                    nearest_network[i,j,0]=new_row
                    nearest_network[i,j,1]=new_cols
                    
    except:
        assert 1==5, "please check the boundaries of your catchment after cropping the catchment using the a polygon it creates anomalies athe boundary "
        
    # calculate the elevation difference between the cell and the nearest drainage cell
    # or height avove nearst drainage
    HAND=np.ones((rows,cols))*np.nan
    
    for i in range(rows):
        for j in range(cols):
            if dem_A[i,j] != no_val:
                HAND[i,j] = dem_A[i,j] - dem_A[int(nearest_network[i,j,0]),int(nearest_network[i,j,1])]
    
    # calculate the distance to the nearest drainage cell using flow path length
    # or distance to nearest drainage
    DTND=np.ones((rows,cols))*np.nan
    
    for i in range(rows):
        for j in range(cols):
            if dem_A[i,j] != no_val:
                DTND[i,j] = fpl_A[i,j] - fpl_A[int(nearest_network[i,j,0]),int(nearest_network[i,j,1])]
    
    return HAND, DTND


def SaveParameters(DistParFn, Raster, Par, No_parameters, snow, kub, klb, 
                   Path=None):
    """
    ============================================================
        SaveParameters(DistParFn, Raster, Par, No_parameters, snow, kub, klb, Path=None)
    ============================================================
    this function takes generated parameters by the calibration algorithm, 
    distributed them with a given function and save them asrasters
    
    Inputs:
    ----------
        1-DistParFn:
            [function] function to distribute the parameters (all functions are
            in Hapi.DistParameters )
        2-Raster:
            [gdal.dataset] raster to get the spatial information
        3-Par
            [list or numpy ndarray] parameters as 1D array or list
        4-no_parameters:
            [int] number of the parameters in the conceptual model
        5-snow:
            [integer] number to define whether to take parameters of 
            the conceptual model with snow subroutine or without
        5-kub:
            [numeric] upper bound for k parameter in muskingum function
        6-klb:
            [numeric] lower bound for k parameter in muskingum function
         7-Path:
             [string] path to the folder you want to save the parameters in
             default value is None (parameters are going to be saved in the
             current directory)
     
    Outputs:
    ----------
         Rasters for parameters of the distributed model
     
   Examples:     
   ----------
        DemPath = path+"GIS/4000/dem4000.tif"
        Raster=gdal.Open(DemPath)
        ParPath = "par15_7_2018.txt"
        par=np.loadtxt(ParPath)
        klb=0.5
        kub=1
        no_parameters=12
        DistParFn=DP.par3dLumped
        Path="parameters/"
        snow=0
        
        SaveParameters(DistParFn, Raster, par, no_parameters,snow ,kub, klb,Path)
    """
    assert callable(DistParFn), " please check the function to distribute your parameters"
    assert type(Raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(Par)==np.ndarray or type(Par)==list, "par_g should be of type 1d array or list"
    assert type(No_parameters) == int, "No of parameters should be integer"
    assert isinstance(kub,numbers.Number) , " kub should be a number"
    assert isinstance(klb,numbers.Number) , " klb should be a number"
    assert type(Path) == str, "path should be of type string"
    assert os.path.exists(Path), Path + " you have provided does not exist"
    
    par2d=DistParFn(Par,Raster,No_parameters,kub,klb)
    # save 
    if snow == 0: # now snow subroutine
        pnme=["01rfcf.tif","02FC.tif", "03BETA.tif", "04ETF.tif", "05LP.tif", "06CFLUX.tif", "07K.tif",
              "08K1.tif","09ALPHA.tif", "10PERC.tif", "11Kmuskingum.tif", "12Xmuskingum.tif"]
    else: # there is snow subtoutine 
        pnme=["01ltt.tif", "02utt.tif", "03rfcf.tif", "04sfcf.tif", "05ttm.tif", "06cfmax.tif", "07cwh.tif",
              "08cfr.tif", "09fc.tif", "10fc.tif", "11beta.tif","12etf.tif","13lp.tif","14cflux.tif",
              "15k.tif","16k1.tif","17alpha.tif","18perc.tif"]
        
    if Path != None:
        pnme=[Path+i for i in pnme]

    for i in range(np.shape(par2d)[2]):
        GIS.RasterLike(Raster,par2d[:,:,i],pnme[i])


def ParametersNO(raster, no_parameters, no_lumped_par,
                 HRUs=0):
    """
    ==================================================================
         ParametersNO(raster,no_parameters,no_lumped_par,HRUs=0)
    ==================================================================
    this function calculates the nomber of parameters that the optimization
    algorithm is going top search for, use it only in case of totally distributed 
    catchment parameters (in case of lumped parameters no of parameters are the same
    as the no of parameters of the conceptual model)
    
    Inputs:
    ----------
        1- raster:
            [gdal.dataset] raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        2- no_parameters
            [int] no of parameters of the cell according to the rainfall runoff model
        3-no_lumped_par:
            [int] nomber of lumped parameters, you have to enter the value of 
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        4-HRUs:
            [0 or 1] 0 to define that no hydrologic response units (HRUs), 1 to define that 
            HRUs are used
    
    """
    # input data validation
    # data type
    assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(no_parameters)== int, "no of lumped parameters should be integer"
    assert type(no_lumped_par)== int, "no of lumped parameters should be integer"
    
    # read the raster    
    raster_A=raster.ReadAsArray()
    # get the shape of the raster
    rows=raster.RasterYSize
    cols=raster.RasterXSize
    # get the no_value of in the raster    
    noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())
    
    # count the number of non-empty cells 
    no_elem = np.size(raster_A[:,:])-np.count_nonzero((raster_A[raster_A==noval])) 
    
    if HRUs == 0:
        # input values
        if no_lumped_par > 0:
            ParametersNO=(no_elem*(no_parameters-no_lumped_par))+no_lumped_par #,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
        else:
            # if there is no lumped parameters
            ParametersNO = no_elem*no_parameters #,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)
    else:
        # count the number of non-empty cells 
        values=list(set([int(raster_A[i,j]) for i in range(rows) for j in range(cols) if raster_A[i,j] != noval]))
        no_elem=len(values)
        
        # input values
        if no_lumped_par > 0:
            ParametersNO = (no_elem*(no_parameters-no_lumped_par))+no_lumped_par #,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
        else:
            # if there is no lumped parameters
            ParametersNO = no_elem*no_parameters #,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)
        
    return ParametersNO