# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:17:54 2018

@author: Mostafa
"""
import numpy as np
import Hapi.hbv_lake as hbv_lake
from Hapi.distrrm import DistributedRRM as distrrm
from Hapi.routing import Routing as routing

class Wrapper():

    def __init__(self):
        pass

    @staticmethod
    def HapiModel(Model, ll_temp=None, q_0=None):
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
            1-statevariables:
                [numpy ndarray] 4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
            2-qlz:
                [numpy ndarray] 3D array of the lower zone discharge
            3-quz:
                [numpy ndarray] 3D array of the upper zone discharge
            4-qout:
                [numpy array] 1D timeseries of discharge at the outlet of the catchment
                of unit m3/sec
            5-quz_routed:
                [numpy ndarray] 3D array of the upper zone discharge  accumulated and
                routed at each time step
            6-qlz_translated:
                [numpy ndarray] 3D array of the lower zone discharge translated at each time step
        """
        # run the rainfall runoff model separately
        distrrm.RunLumpedRRM(Model)

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        Model.qout = Model.qout[:-1]



    @staticmethod
    def HapiWithlake(Model, Lake,ll_temp=None, q_0=None):

        plake = Lake.MeteoData[:,0]
        et = Lake.MeteoData[:,1]
        t = Lake.MeteoData[:,2]
        tm = Lake.MeteoData[:,3]

        # lake simulation
        Lake.Qlake, _ = hbv_lake.simulate(plake, t, et, Lake.Parameters,
                                      [Model.Timef, Lake.CatArea, Lake.LakeArea],
                                      Lake.StageDischargeCurve, 0,
                                      init_st=Lake.InitialCond,
                                      ll_temp=tm, lake_sim=True)
        # qlake is in m3/sec
        # lake routing
        Lake.QlakeR = routing.muskingum(Lake.Qlake, Lake.Qlake[0], Lake.Parameters[11],
                                  Lake.Parameters[12], Model.Timef)

        # subcatchment
        # st, q_lz, q_uz = distrrm.RunLumpedRRM(Model)
        distrrm.RunLumpedRRM(Model)

        # routing lake discharge with DS cell k & x and adding to cell Q
        q_lake = routing.muskingum(Lake.QlakeR,Lake.QlakeR[0],
                                   Model.Parameters[Lake.OutflowCell[0],Lake.OutflowCell[1],10],
                                   Model.Parameters[Lake.OutflowCell[0],Lake.OutflowCell[1],11],
                                   Model.Timef)

        q_lake = np.append(q_lake,q_lake[-1])
        # both lake & Quz are in m3/s
        Model.q_uz[Lake.OutflowCell[0],Lake.OutflowCell[1],:] = Model.q_uz[Lake.OutflowCell[0],Lake.OutflowCell[1],:] + q_lake

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        Model.q_out = Model.q_out[:-1]

        # return q_out, q_uz_routed, q_lz_trans


    @staticmethod
    def FW1Withlake(ConceptualModel,FPL,sp_prec,sp_et,sp_temp,
                     parameters,p2,snow,init_st,lakeCalibArray,StageDischargeCurve,
                     LakeParameters,lakecell,lake_initial,ll_temp=None, q_0=None):

        plake = lakeCalibArray[:,0]
        et = lakeCalibArray[:,1]
        t = lakeCalibArray[:,2]
        tm = lakeCalibArray[:,3]

        # lake simulation
        q_lake, _ = hbv_lake.simulate(plake, t, et, LakeParameters[:-1], p2,StageDischargeCurve,
                                      0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
        # qlake is in m3/sec
        # lake routing
        qlake_r = routing.TriangularRouting(q_lake,LakeParameters[-1])

        # subcatchment
        AdditionalParameters = p2[0:2]
        st, q_lz, q_uz = distrrm.RunLumpedRRM(ConceptualModel, FPL, sp_prec=sp_prec,
                                              sp_et=sp_et, sp_temp=sp_temp, sp_pars=parameters,
                                              p2=AdditionalParameters, snow=snow,
                                              init_st=init_st)

        SPMAXBAS = parameters[:,:,-1]
        q_uz = distrrm.DistMAXBAS(FPL, SPMAXBAS, q_uz)
        #
        q_lz1 = np.array([np.nansum(q_lz[:,:,i]) for i in range(sp_prec.shape[2]+1)]) # average of all cells (not routed mm/timestep)
        q_uz1 = np.array([np.nansum(q_uz[:,:,i]) for i in range(sp_prec.shape[2]+1)]) # average of all cells (routed mm/timestep)

        q_out = (q_lz1 + q_uz1) * p2[1] / (p2[0] * 3.6)

        q_out = q_out[:-1] + qlake_r

        return st, q_out, q_uz, q_lz

    @staticmethod
    def Lumped(Model, Routing=0, RoutingFn=[]):
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
        # assert type(data) == np.ndarray , "meteorological data should be entered in array "
        assert callable(RoutingFn) , "routing function should be of type callable (function that takes arguments)"

        # data
        p = Model.data[:,0]
        et = Model.data[:,1]
        t = Model.data[:,2]
        tm = Model.data[:,3]

        # from the conceptual model calculate the upper and lower response mm/time step
        q_uz, q_lz, st = Model.LumpedModel.Simulate(p, t, et, Model.Parameters,
                                                     init_st = Model.InitialCond,
                                                     ll_temp = tm,
                                                     q_init = None,
                                                     snow = Model.Snow)
        # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        q_uz = q_uz*Model.CatArea/(Model.Timef*3.6)
        q_lz = q_lz*Model.CatArea/(Model.Timef*3.6)

        q_sim = q_uz + q_lz

        if Routing != 0 :
            q_sim = RoutingFn(np.array(q_sim[:-1]), Model.Parameters[-1])

        return st, q_sim