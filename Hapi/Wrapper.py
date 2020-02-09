# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:17:54 2018

@author: Mostafa
"""
#%links


#%library
import numpy as np
import gdal
from types import ModuleType

# functions
#import DistParameters
import Hapi.HBV_Lake as HBV_Lake
import Hapi.DistRRM as DistRRM
import Hapi.Routing as Routing



def HAPIWithlake(ConceptualModel,flow_acc,flow_direct,sp_prec,sp_et,sp_temp,
                 parameters,p2,snow,init_st,lakeCalibArray,StageDischargeCurve,
                 LakeParameters,lakecell,lake_initial,ll_temp=None, q_0=None):

    plake = lakeCalibArray[:,0]
    et = lakeCalibArray[:,1]
    t = lakeCalibArray[:,2]
    tm = lakeCalibArray[:,3]

    # lake simulation
    q_lake, _ = HBV_Lake.simulate(plake, t, et, LakeParameters, p2,StageDischargeCurve,
                                  0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # qlake is in m3/sec
    # lake routing
    qlake_r = Routing.muskingum(q_lake,q_lake[0],LakeParameters[11],
                              LakeParameters[12],p2[0])

    # subcatchment
    AdditionalParameters = p2[0:2]
    st, q_lz, q_uz = DistRRM.RunLumpedRRP(ConceptualModel,flow_acc, sp_prec=sp_prec,
                                          sp_et=sp_et, sp_temp=sp_temp, sp_pars=parameters,
                                          p2=AdditionalParameters, snow=snow,
                                          init_st=init_st)

    # routing lake discharge with DS cell k & x and adding to cell Q
    q_lake = Routing.muskingum(qlake_r,qlake_r[0],parameters[lakecell[0],lakecell[1],10],parameters[lakecell[0],lakecell[1],11],p2[0])
    q_lake = np.append(q_lake,q_lake[-1])
    # both lake & Quz are in m3/s
    q_uz[lakecell[0],lakecell[1],:] = q_uz[lakecell[0],lakecell[1],:] + q_lake

    # run the GIS part to rout from cell to another
    q_out, q_uz_routed, q_lz_trans = DistRRM.SpatialRouting(q_lz, q_uz,flow_acc,flow_direct,parameters,p2)

    q_out=q_out[:-1]

    return st, q_out, q_uz_routed, q_lz_trans


def HAPIModel(ConceptualModel,flow_acc,flow_direct,sp_prec,sp_et,sp_temp,sp_par,p2, snow,
                    init_st,ll_temp=None, q_0=None):
    """
    =======================================================================
      Dist_model(DEM,flow_acc,flow_direct,sp_prec,sp_et,sp_temp,sp_par,p2,kub,klb,init_st,ll_temp,q_0)
    =======================================================================
    this wrapper function connects all components of the model:
        1- rainfall runoff model runs separately for each cell
        2- GIS routing scheme (routing is following river network)

    Inputs:
    ----------
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
    ----------
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



def FW1Withlake(ConceptualModel,FPL,sp_prec,sp_et,sp_temp,
                 parameters,p2,snow,init_st,lakeCalibArray,StageDischargeCurve,
                 LakeParameters,lakecell,lake_initial,ll_temp=None, q_0=None):

    plake = lakeCalibArray[:,0]
    et = lakeCalibArray[:,1]
    t = lakeCalibArray[:,2]
    tm = lakeCalibArray[:,3]

    # lake simulation
    q_lake, _ = HBV_Lake.simulate(plake, t, et, LakeParameters[:-1], p2,StageDischargeCurve,
                                  0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # qlake is in m3/sec
    # lake routing
    qlake_r = Routing.TriangularRouting(q_lake,LakeParameters[-1])

    # subcatchment
    AdditionalParameters = p2[0:2]
    st, q_lz, q_uz = DistRRM.RunLumpedRRP(ConceptualModel, FPL, sp_prec=sp_prec,
                                          sp_et=sp_et, sp_temp=sp_temp, sp_pars=parameters,
                                          p2=AdditionalParameters, snow=snow,
                                          init_st=init_st)

    SPMAXBAS = parameters[:,:,-1]
    q_uz = DistRRM.DistMAXBAS(FPL, SPMAXBAS, q_uz)
    #
    q_lz1 = np.array([np.nansum(q_lz[:,:,i]) for i in range(sp_prec.shape[2]+1)]) # average of all cells (not routed mm/timestep)
    q_uz1 = np.array([np.nansum(q_uz[:,:,i]) for i in range(sp_prec.shape[2]+1)]) # average of all cells (routed mm/timestep)

    q_out = (q_lz1 + q_uz1) * p2[1] / (p2[0] * 3.6)

    q_out = q_out[:-1] + qlake_r

    return st, q_out, q_uz, q_lz


def Lumped(ConceptualModel,data,parameters,p2,init_st,snow,Routing=0, RoutingFn=[]):
    """
    ==========================================================
        Lumped(ConceptualModel,data,parameters,p2,snow,Routing=None, RoutingFn=[])
    ==========================================================

    Inputs:
    ----------
        1-ConceptualModel:
            [function] conceptual model and it should contain a function called simulate
        2-data:
            [numpy array] meteorological data as array with the first column as precipitation
            second as evapotranspiration, third as temperature and forth column as
            long term average temperature
        3- parameters:
            [numpy array] conceptual model parameters as array
        4-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
        5-init_st:
            [list] initial state variables values [sp, sm, uz, lz, wc].
        6-Routing:
            [0 or 1] to decide wether t route the generated discharge hydrograph or not
        7-RoutingFn:
            [function] function to route the dischrge hydrograph.

    Outputs:
    ----------
        1- st:
            [numpy array] 3d array of the 5 state variable data for each cell
        2- q_lz:
            [numpy array] 1d array of the calculated discharge.

    Examples:
    ----------
        p2=[24, 1530]
        #[sp,sm,uz,lz,wc]
        init_st=[0,5,5,5,0]
        snow=0
    """
    ### input data validation
    # data type
    assert isinstance(ConceptualModel,ModuleType) , "ConceptualModel should be a module or a python file contains functions "
    assert type(data) == np.ndarray , "meteorological data should be entered in array "
    assert callable(RoutingFn) , "routing function should be of type callable (function that takes arguments)"
    assert np.shape(data)[1] == 3 or np.shape(data)[1] == 4," meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "

    # data
    p=data[:,0]
    et=data[:,1]
    t=data[:,2]
    tm=data[:,3]

    # from the conceptual model calculate the upper and lower response mm/time step

    q_uz, q_lz, st = ConceptualModel.Simulate(p, t, et, parameters, p2,
                                                 init_st = init_st,
                                                 ll_temp = tm,
                                                 q_init = None,
                                                 snow = 0)
    q_uz = q_uz*p2[1]/(p2[0]*3.6) # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
    q_lz = q_lz*p2[1]/(p2[0]*3.6) # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25

    q_sim = q_uz + q_lz

    if Routing != 0 :
        q_sim=RoutingFn(np.array(q_sim[:-1]), parameters[-1])

    return st, q_sim