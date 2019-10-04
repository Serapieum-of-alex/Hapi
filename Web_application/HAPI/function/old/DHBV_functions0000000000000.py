# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:21:41 2018

@author: Mostafa
"""
#%% library
import pickle
import numpy as np



#%% functions
def save_obj(obj, saved_name ):
    with open( saved_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(saved_name ):
    with open( saved_name + '.pkl', 'rb') as f:
        return pickle.load(f)




def get_raster_data(dem):
    """
    _get_mask(dem)
    to create a mask by knowing the stored value inside novalue cells 
    Inputs:
        1- flow path lenth raster
    Outputs:
        1- mask:array with all the values in the flow path length raster
        2- no_val: value stored in novalue cells
    """
    no_val = np.float32(dem.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
    mask = dem.ReadAsArray() # read all values
    return mask, no_val


def calculateK(x,position,UB,LB):
    '''
    calculateK(x,position,UB,LB):
        this function takes value of x parameter and generate 100 random value of k parameters between
        upper & lower constraint then the output will be the value coresponding to the giving position
        
        Inputs:
            1- x weighting coefficient to determine the linearity of the water surface
                (one of the parameters of muskingum routing method)
            2- position 
                random position between upper and lower bounds of the k parameter
            3-UB 
                upper bound for k parameter
            3-LB 
                Lower bound for k parameter
    '''
    constraint1=0.5*1/(1-x) # k has to be smaller than this constraint
    constraint2=0.5*1/x   # k has to be greater than this constraint
    
    if constraint2 >= UB : #if constraint is higher than UB take UB
        constraint2 =UB
        
    if constraint1 <= LB : #if constraint is lower than LB take UB
        constraint1 =LB
    
    generatedK=np.linspace(constraint1,constraint2,101)
    k=generatedK[int(round(position,0))]
    return k


def par2d_lumpedK1(par_g,raster,no_parameters,no_parameters_lake,kub,klb):
    '''
    par2d_lumpedK1(par_g,flp,no_parameters,no_parameters_lake,kub,klb)
    this function takes a list of parameters and distribute them horizontally on number of cells
    given by a raster 
    
    example a list of 155 value,all parameters are distributed except lower zone coefficient
    (is written at the end of the list) each cell(14 cells) has 11 parameter plus lower zone
    (12 parameters) function will take each 11 parameter and assing them to a specific cell
    then assign the last value (lower zone parameter) to all cells
    14*11=154 + 1 = 155
    Inputs 
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
    Output
        1- par_2d: 3D array of the parameters distributed horizontally on the cells
        2- lake_par: list of the lake parameters
    '''
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
    # take the parameters and assign them to each cell
    for i in range(no_elem):
        par_arr[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]
    # create a list with the value of the lumped parameter(k1) (stored at the end of the list of the parameters)
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
#    par_3d[:,:,-1][f !=no_val]=par_g[(np.shape(par_arr)[0]*np.shape(par_arr)[1])]
    
#    lake_par=par_g[len(par_g)-13:]
    # lake parameters        
    lake_par=par_g[len(par_g)-no_parameters_lake:]
    lake_par[-2]=calculateK(lake_par[-1],lake_par[-2],kub,klb)
    
    return par_2d,lake_par
