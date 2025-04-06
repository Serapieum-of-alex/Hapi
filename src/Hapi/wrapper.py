"""Created on Sun Apr 29 17:17:54 2018.

@author: Mostafa
"""

import numpy as np

from Hapi.routing import Routing as routing
from Hapi.rrm import hbv_lake
from Hapi.rrm.distrrm import DistributedRRM as distrrm


class Wrapper:
    """Wrapper.

    The class connect different commponent together (lumped run of the distributed with the spatial routing) for Hapi
    and for FW1.

    Methods
    -------
        1- HapiModel
        2- RRMWithlake
        3- FW1
        4- FW1Withlake
        5- Lumped
    """

    def __init__(self):
        """Run the whole setup."""
        pass

    @staticmethod
    def RRMModel(Model, ll_temp=None, q_0=None):
        """RRMModel.

            HapiModel connect two modules :
            1- The distributed rainfall runoff: model runs separately for each cell
            2- The Spatial routing scheme (routing is following river network)

        Parameters
        ----------
        Model
            DEM: [gdal.dataset]
                DEM raster file of the catchment (clipped to the catchment only)
            flow_acc: [gdal.dataset]
                flow accumulation raster file of the catchment (clipped to the catchment only)
            flow_direct: [gdal.dataset]
                flow Direction raster file of the catchment (clipped to the catchment only)
            sp_prec: [numpy array]
                3d array of the precipitation data, sp_prec should
                have the same 2d dimension of raster input
            sp_et: [numpy array]
                3d array of the evapotranspiration data, sp_et should
                have the same 2d dimension of raster input
            sp_temp: [numpy array]
                3d array of the temperature data, sp_temp should
                have the same 2d dimension of raster input
            sp_par: [numpy array]
                number of 2d arrays of the catchment properties spatially
                distributed in 2d and the third dimension is the number of parameters,
                sp_pars should have the same 2d dimension of raster input
            p2: [List]
                list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            kub: [float]
                upper bound of K value (traveling time in muskingum routing method)
            klb: [float]
                Lower bound of K value (traveling time in muskingum routing method)
            init_st: [list]
                initial state variables values [sp, sm, uz, lz, wc]. default=None
            ll_temp: [numpy array]
                3d array of the long term average temperature data
            q_0: [float]
                initial discharge m3/s

        Returns
        -------
        state_variables:
            [numpy ndarray] 4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
        qlz:
            [numpy ndarray] 3D array of the lower zone discharge
        quz:
            [numpy ndarray] 3D array of the upper zone discharge
        qout:
            [numpy array] 1D timeseries of discharge at the outlet of the catchment
            of unit m3/sec
        quz_routed:
            [numpy ndarray] 3D array of the upper zone discharge  accumulated and
            routed at each time step
        qlz_translated:
            [numpy ndarray] 3D array of the lower zone discharge translated at each time step
        """
        # run the rainfall runoff model separately
        distrrm.run_lumped_model(Model)

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def RRMWithlake(Model, Lake, ll_temp=None, q_0=None):
        """RRMWithlake.

            RRMWithlake connects three modules the lake, the distributed
            ranfall-runoff module and spatial routing module

        Parameters
        ----------
        Model : [Catchment object]
            DESCRIPTION.
        Lake : TYPE
            DESCRIPTION.
        ll_temp : TYPE, optional
            DESCRIPTION. The default is None.
        q_0 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """

        plake = Lake.MeteoData[:, 0]
        et = Lake.MeteoData[:, 1]
        t = Lake.MeteoData[:, 2]
        tm = Lake.MeteoData[:, 3]

        # lake simulation
        Lake.Qlake, _ = hbv_lake.simulate(
            plake,
            t,
            et,
            Lake.Parameters,
            [Model.conversion_factor, Lake.CatArea, Lake.LakeArea],
            Lake.StageDischargeCurve,
            0,
            init_st=Lake.InitialCond,
            ll_temp=tm,
            lake_sim=True,
        )
        # qlake is in m3/sec
        # lake routing
        Lake.QlakeR = routing.Muskingum_V(
            Lake.Qlake,
            Lake.Qlake[0],
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        # routing lake discharge with DS cell k & x and adding to cell Q
        qlake = routing.Muskingum_V(
            Lake.QlakeR,
            Lake.QlakeR[0],
            Model.Parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 10],
            Model.Parameters[Lake.OutflowCell[0], Lake.OutflowCell[1], 11],
            Model.conversion_factor,
        )

        qlake = np.append(qlake, qlake[-1])
        # both lake & Quz are in m3/s
        Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] = (
            Model.quz[Lake.OutflowCell[0], Lake.OutflowCell[1], :] + qlake
        )

        # run the GIS part to rout from cell to another
        distrrm.SpatialRouting(Model)

        # Model.qout = Model.qout[:-1]

    @staticmethod
    def FW1(Model, ll_temp=None, q_0=None):
        """FW1.

        FW1 connects two module :
            1- The distributed rainfall-runoff module
            2- Triangular function-1 routing method
        Parameters
        ----------
        Model : TYPE
            DESCRIPTION.
        ll_temp : TYPE, optional
            DESCRIPTION. The default is None.
        q_0 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """

        # subcatchment
        distrrm.run_lumped_model(Model)

        distrrm.DistMaxbas1(Model)

        qlz1 = np.array(
            [np.nansum(Model.qlz[:, :, i]) for i in range(Model.TS)]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [np.nansum(Model.quz[:, :, i]) for i in range(Model.TS)]
        )  # average of all cells (routed mm/timestep)

        Model.qout = qlz1 + quz1

        Model.qout = Model.qout[:-1]

    @staticmethod
    def FW1Withlake(Model, Lake, ll_temp=None, q_0=None):
        """FW1Withlake.

        FW1 connects two module :
            1- The distributed rainfall-runoff module
            2- Triangular function-1 routing method
            3- Lake module

        Parameters
        ----------
        Model : TYPE
            DESCRIPTION.
        Lake : TYPE
            DESCRIPTION.
        ll_temp : TYPE, optional
            DESCRIPTION. The default is None.
        q_0 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """

        plake = Lake.MeteoData[:, 0]
        et = Lake.MeteoData[:, 1]
        t = Lake.MeteoData[:, 2]
        tm = Lake.MeteoData[:, 3]

        # lake simulation
        Lake.Qlake, _ = hbv_lake.simulate(
            plake,
            t,
            et,
            Lake.Parameters,
            [Model.conversion_factor, Lake.CatArea, Lake.LakeArea],
            Lake.StageDischargeCurve,
            0,
            init_st=Lake.InitialCond,
            ll_temp=tm,
            lake_sim=True,
        )

        # qlake is in m3/sec
        # lake routing
        Lake.QlakeR = routing.muskingum(
            Lake.Qlake,
            Lake.Qlake[0],
            Lake.Parameters[11],
            Lake.Parameters[12],
            Model.conversion_factor,
        )

        # subcatchment
        distrrm.run_lumped_model(Model)

        distrrm.DistMAXBAS(Model)

        qlz1 = np.array(
            [
                np.nansum(Model.qlz[:, :, i])
                for i in range(Model.Parameters.shape[2] + 1)
            ]
        )  # average of all cells (not routed mm/timestep)
        quz1 = np.array(
            [
                np.nansum(Model.quz[:, :, i])
                for i in range(Model.Parameters.shape[2] + 1)
            ]
        )  # average of all cells (routed mm/timestep)

        qout = qlz1 + quz1

        # qout = (qlz1 + quz1) * Model.CatArea / (Model.conversion_factor* 3.6)

        Model.qout = qout[:-1] + Lake.QlakeR

    @staticmethod
    def Lumped(Model, Routing=0, RoutingFn=[]):
        """Lumped.

        Parameters
        ----------
        Model: [function]
            conceptual model and it should contain a function called simulate
            data: [numpy array]
                meteorological data as array with the first column as precipitation
                second as evapotranspiration, third as temperature and forth column as
                long term average temperature
            parameters: [numpy array]
                conceptual model parameters as array
            p2: [List]
                list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            init_st: [list]
                initial state variables values [sp, sm, uz, lz, wc].
        Routing: [0 or 1]
            to decide wether t route the generated discharge hydrograph or not
        RoutingFn: [function]
            function to route the dischrge hydrograph.

        Returns
        -------
        st: [numpy array]
            3d array of the 5 state variable data for each cell
        q_lz: [numpy array]
            1d array of the calculated discharge.

        Examples
        --------
        >>> p2=[24, 1530]
        >>> #[sp,sm,uz,lz,wc]
        >>> init_st=[0,5,5,5,0]
        >>> snow=0
        """
        ### input data validation
        assert callable(
            RoutingFn
        ), "routing function should be of type callable (function that takes arguments)"

        # data
        p = Model.data[:, 0]
        et = Model.data[:, 1]
        t = Model.data[:, 2]
        tm = Model.data[:, 3]

        # from the conceptual model calculate the upper and lower response mm/time step
        Model.quz, Model.qlz, Model.state_variables = Model.LumpedModel.simulate(
            p,
            t,
            et,
            tm,
            Model.Parameters,
            init_st=Model.InitialCond,
            q_init=Model.q_init,
            snow=Model.Snow,
        )
        # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
        # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
        Model.quz = Model.quz * Model.CatArea / Model.conversion_factor
        Model.qlz = Model.qlz * Model.CatArea / Model.conversion_factor

        Model.Qsim = Model.quz + Model.qlz

        if Routing != 0 and Model.Maxbas:
            Model.Qsim = RoutingFn(np.array(Model.Qsim[:-1]), Model.Parameters[-1])
        elif Routing != 0:
            Model.Qsim = RoutingFn(
                np.array(Model.Qsim[:-1]),
                Model.Qsim[0],
                Model.Parameters[-2],
                Model.Parameters[-1],
                Model.dt,
            )


if __name__ == "__main__":
    print("Wrapper")
