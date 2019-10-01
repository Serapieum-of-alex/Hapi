# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:17:54 2018

@author: Mostafa
"""
#%links


#%library
import numpy as np
import gdal

# functions
import DistParameters
import HBV_Lake
import DistRRM
import Routing
#import Performance_criteria

def Dist_model_lake(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial,lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
#    Qobs=data[:,4]
    
    # distribute the parameters to a 2d array
    jiboa_par,lake_par=DistParameters.par2d_lumpedK1_lake(sp_pars,DEM,12,13,kub,klb)
    
    
    # lake simulation
    q_lake, _ = HBV_Lake.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=Routing.muskingum(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz,_= DistRRM.Dist_HBV2(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
#    RMSEE=Performance_criteria.rmse(Qobs,q_tot)
    
    return q_tot,q_lake,q_uz_routed, q_lz


def Dist_model(ConceptualModel,flow_acc,flow_direct,sp_prec,sp_et,sp_temp,sp_par,p2, snow,
                    init_st,ll_temp=None, q_0=None):
    """
    =======================================================================
      Dist_model(DEM,flow_acc,flow_direct,sp_prec,sp_et,sp_temp,sp_par,p2,kub,klb,init_st,ll_temp,q_0)
    =======================================================================
    this wrapper function connects all components of the model:
        1- rainfall runoff model runs separately for each cell
        2- GIS routing scheme (routing is following river network)
    
    Inputs:
        1-DEM:
            [gdal.dataset] DEM raster file of the catchment (clipped to the catchment only)
        2-flow_acc:
            [gdal.dataset] flow accumulation raster file of the catchment (clipped to the catchment only)
        3-flow_direct:
            [gdal.dataset] flow Direction raster file of the catchment (clipped to the catchment only)
        4-sp_prec:
            [numpy array] 3d array of the precipitation data, sp_prec should
            have the same 2d dimension of raster input
        5-sp_et:
            [numpy array] 3d array of the evapotranspiration data, sp_et should
            have the same 2d dimension of raster input
        6-sp_temp:
            [numpy array] 3d array of the temperature data, sp_temp should
            have the same 2d dimension of raster input
        7-sp_par:
            [numpy array] number of 2d arrays of the catchment properties spatially 
            distributed in 2d and the third dimension is the number of parameters,
            sp_pars should have the same 2d dimension of raster input
        8-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
        9-kub:
            [float] upper bound of K value (traveling time in muskingum routing method)
        10-klb:
            [float] Lower bound of K value (traveling time in muskingum routing method)
        11-init_st:
            [list] initial state variables values [sp, sm, uz, lz, wc]. default=None
        12-ll_temp:
            [numpy array] 3d array of the long term average temperature data
        13-q_0:
            [float] initial discharge m3/s
    Outputs:
        1- st:
            [numpy array] 3d array of the 5 state variable data for each cell
        2- q_lz:
            [numpy array] 1d array of the calculated discharge of the lower zone
            of the rainfall runoff model
        3- q_uz:
            [numpy array] 3d array of calculated discharge for each cell for the 
            entire time series
    """
    ### input data validation
    # data type
#    assert type(DEM)==gdal.Dataset, "DEM should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(flow_acc)==gdal.Dataset, "flow_acc should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(flow_direct)==gdal.Dataset, "flow_direct should be read using gdal (gdal dataset please read it using gdal library) "
    
    # input dimensions
#    [rows,cols]=DEM.ReadAsArray().shape
#    [acc_rows,acc_cols]=flow_acc.ReadAsArray().shape
    [rows,cols]=flow_acc.ReadAsArray().shape
    [fd_rows,fd_cols]=flow_direct.ReadAsArray().shape
#    assert acc_rows == rows and fd_rows == rows and acc_cols == cols and fd_cols == cols, "all input data should have the same number of rows"
    assert fd_rows == rows and fd_cols == cols, "all input data should have the same number of rows"
    
    # input values
    # check flow accumulation input raster
    acc_noval=np.float32(flow_acc.GetRasterBand(1).GetNoDataValue())
    acc=flow_acc.ReadAsArray()
    no_elem = np.size(acc[:,:])-np.count_nonzero((acc[acc==acc_noval]))
    acc_val=[acc[i,j] for i in range(rows) for j in range(cols) if acc[i,j] != acc_noval]
    acc_val=list(set(acc_val))
    acc_val_mx=max(acc_val)
    assert acc_val_mx == no_elem or acc_val_mx == no_elem -1, "flow accumulation raster values are not correct max value should equal number of cells or number of cells -1"
    
    # check flow direction input raster
    fd_noval=np.float32(flow_direct.GetRasterBand(1).GetNoDataValue())
    fd=flow_direct.ReadAsArray()
    fd_val=[int(fd[i,j]) for i in range(rows) for j in range(cols) if fd[i,j] != fd_noval]
    fd_val=list(set(fd_val))
    fd_should=[1,2,4,8,16,32,64,128]
    assert all(fd_val[i] in fd_should for i in range(len(fd_val))), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "
    
    # run the rainfall runoff model separately 
    st, q_lz, q_uz = DistRRM.RunLumpedRRP(ConceptualModel,flow_acc, sp_prec=sp_prec, sp_et=sp_et, #DEM
                                          sp_temp=sp_temp, sp_pars=sp_par, p2=p2, snow=snow,
                                          init_st=init_st)
    # run the GIS part to rout from cell to another
    q_out, q_uz_routed, q_lz_trans = DistRRM.SpatialRouting(q_lz, q_uz,flow_acc,flow_direct,sp_par,p2)
    
    q_out=q_out[:-1]
    
    return st, q_out, q_uz_routed, q_lz_trans