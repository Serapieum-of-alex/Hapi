"""Calibration.

calibration contains functions to to connect the parameter spatial distribution
function with the with both component of the spatial representation of the hydrological
process (conceptual model & spatial routing) to calculate the performance of predicted
runoff at known locations based on given performance function

@author: Mostafa
"""
from typing import Any, Optional, List

import numpy as np
from Oasis.harmonysearch import HSapi
from Oasis.optimization import Optimization

from Hapi.catchment import Catchment
from Hapi.wrapper import Wrapper


class Calibration(Catchment):
    """Calibration.

    Calibration class contains to connect the parameter spatial distribution
    function with both components of the spatial representation of the
    hydrological process (conceptual model & spatial routing) to calculate the
    performance of predicted runoff at known locations based on given
    performance function

    The calibration class is subclass from the Catchment superclass, so you
    need to create the Catchment object first to be able to run the calibration
    """

    def __init__(
        self,
        name: Any,
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        spatial_resolution: Optional[str] = "Lumped",
        temporal_resolution: Optional[str] = "Daily",
        routing_method: Optional[str] = "Muskingum",
    ):
        """Calibration.

        to instantiate the Calibration object, you need to provide the following
        arguments

        Parameters
        ----------
        name : [str]
            Name of the Catchment.
        start : [str]
            starting date.
        end : [str]
            end date.
        fmt: [str], optional
            format of the given date. The default is "%Y-%m-%d".
        spatial_resolution: [str], optional
            Lumped or 'Distributed'. The default is 'Lumped'.
        temporal_resolution: [str], optional
            "Hourly" or "Daily". The default is "Daily".
        """
        super().__init__(
            name,
            start,
            end,
            fmt,
            spatial_resolution,
            temporal_resolution,
            routing_method,
        )

    def read_objective_function(self, objective_function: callable, args):
        """read_objective_function.

        read_objective_function method takes the objective function and any arguments
        that are needed to be passed to the objective function.

        Parameters
        ----------
        objective_function : [function]
            callable function to calculate any kind of metric to be used in the
            calibration.
        args: [positional/keyword arguments]
            any kind of argument you want to pass to your objective function.

        Returns
        -------
        None.
        """
        # check objective_function
        assert callable(
            objective_function
        ), "The Objective function should be a function"
        self.objective_function = objective_function

        if args is None:
            args = []

        self.OFArgs = args

        print("Objective function is read successfully")

    def extract_discharge(self, factor: List = None):
        """extractDischarge.

        extractDischarge method extracts the discharge hydrograph in the
        Q

        Parameters
        ----------
        factor: [list/None]
            list of factor if you want to multiply the simulated discharge by
            a factor you have to provide a list of the factor (as many factors
            as the number of gauges). The default is False.

        Returns
        -------
        None.
        """
        self.Qsim = np.zeros((self.TS - 1, len(self.GaugesTable)))
        # error = 0
        for i in range(len(self.GaugesTable)):
            Xind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_row"])
            Yind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_col"])
            # gaugeid = self.GaugesTable.loc[self.GaugesTable.index[i],"id"]

            # Quz = self.quz_routed[Xind,Yind,:-1]
            # Qlz = self.qlz_translated[Xind,Yind,:-1]
            # self.Qsim[:,i] = Quz + Qlz

            Qsim = np.reshape(self.Qtot[Xind, Yind, :-1], self.TS - 1)

            if factor is not None:
                self.Qsim[:, i] = Qsim * factor[i]
            else:
                self.Qsim[:, i] = Qsim

            # Qobs = Coello.QGauges.loc[:,gaugeid]
            # error = error + objective_function(Qobs, Qsim)

        # return error

    def run_calibration(self, SpatialVarFun, OptimizationArgs, printError=None):
        """Run Calibration.

            - This function runs the calibration algorithm for the conceptual distributed
            hydrological model

        Parameters
        ----------
        SpatialVarFun: [function]

        OptimizationArgs: [Dict]

        printError: [bool]
            Default is None.

        Parameters that should be defined before running the function.
            ConceptualModel:
                [function] conceptual model and it should contain a function called simulate

            Basic_inputs:
                1-p2:
                    [List] list of unoptimized parameters
                    p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                    p2[1] = catchment area in km2
                2-init_st:
                    [list] initial values for the state variables [sp,sm,uz,lz,wc] in mm
                3-UB:
                    [Numeric] upper bound of the values of the parameters
                4-LB:
                    [Numeric] Lower bound of the values of the parameters
            Q_obs:
                [Numeric] Observed values of discharge

            lumpedParNo:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            lumpedParPos:
                [List] list of order or position of the lumped parameter among all
                the parameters of the lumped model (order starts from 0 to the length
                of the model parameters), default is [] (empty), the following order
                of parameters is used for the lumped HBV model used
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
                c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            objective_function:
                [function] objective function to calculate the performance of the model
                and to be used in the calibration
            *args:
                other arguments needed on the objective function

        Returns
        -------
        st: [4D array]
            state variables
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment
        q_uz: [3D array]
            Distributed discharge for each cell

        Example:
        ----------
        >>> prec_path = "meteodata/4000/calib/prec"
        >>> evap_path = "meteodata/4000/calib/evap"
        >>> temp_path = "meteodata/4000/calib/temp"
        >>> flow_acc_path = "GIS/4000/acc4000.tif"
        >>> flow_direction_path = "GIS/4000/fd4000.tif"
        >>> ParPath = "meteodata/4000/parameters.txt"
        >>> p2 = [1, 227.31]
        >>> st, q_out, q_uz_routed = RunModel(prec_path, evap_path, temp_path, DemPath,
        >>>                                   flow_acc_path,flow_direction_path,ParPath,p2)
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        [fd_rows, fd_cols] = self.FlowDirArr.shape
        assert (
            fd_rows == self.rows and fd_cols == self.cols
        ), "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in Basic_inputs.keys() for i in range(4)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) is dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) is dict, "history_fname should be of type string "

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            try:
                # distribute the parameters
                SpatialVarFun.Function(
                    par
                )  # , kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb
                self.Parameters = SpatialVarFun.Par3d
                # run the model
                Wrapper.RRMModel(self)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.QGauges, *[self.GaugesTable]
                    )  # self.qout, self.quz_routed, self.qlz_translated,
                    f = list(range(9, len(par), SpatialVarFun.no_parameters))
                    g = list()
                    for i in range(len(f)):
                        k = par[f[i]]
                        x = par[f[i] + 1]
                        g.append(2 * k * x / self.dt)
                        g.append((2 * k * (1 - x)) / self.dt)

                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except:
                error = np.nan
                g = []
                fail = 1

            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        for i in range(len(self.LB)):
            opt_prob.addVar(
                "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
            )

        opt_prob.addObj("f")

        for i in range(SpatialVarFun.no_elem):
            opt_prob.addCon("g" + str(i) + "-1", "i")
            opt_prob.addCon("g" + str(i) + "-2", "i")

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res

    def FW1Calibration(self, SpatialVarFun, OptimizationArgs, printError=None):
        """FW1Calibration.

        this function runs the calibration algorithm for the conceptual distributed
        hydrological model

        Parameters
        ----------
        ConceptualModel: [function]
            conceptual model and it should contain a function called simulate

        2-Basic_inputs:
            1-p2: [List]
                list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            2-init_st: [list]
                initial values for the state variables [sp,sm,uz,lz,wc] in mm
            3-UB: [Numeric]
                upper bound of the values of the parameters
            4-LB: [Numeric]
                Lower bound of the values of the parameters
        3-Q_obs: [Numeric]
            Observed values of discharge

        6-lumpedParNo: [int]
            nomber of lumped parameters, you have to enter the value of
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        7-lumpedParPos: [List]
            list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        8-objective_function: [function]
            objective function to calculate the performance of the model
            and to be used in the calibration
        9-*args:
            other arguments needed on the objective function

        Returns
        -------
        st: [4D array]
            state variables
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment
        q_uz: [3D array]
            Distributed discharge for each cell

        Example
        -------
        prec_path = prec_path="meteodata/4000/calib/prec"
        evap_path = evap_path="meteodata/4000/calib/evap"
        temp_path = temp_path="meteodata/4000/calib/temp"
        flow_acc_path = "GIS/4000/acc4000.tif"
        flow_direction_path = "GIS/4000/fd4000.tif"
        ParPath = "meteodata/4000/"+"parameters.txt"
        p2=[1, 227.31]
        st, q_out, q_uz_routed = RunModel(prec_path,evap_path,temp_path,DemPath,
                                          flow_acc_path,flow_direction_path,ParPath,p2)
        """
        # input dimensions
        # [rows,cols] = self.FlowAcc.ReadAsArray().shape
        # [fd_rows,fd_cols] = self.FlowDirArr.shape
        # assert fd_rows == self.rows and fd_cols == self.cols, "all input data should have the same number of rows"

        # input dimensions
        assert (
            np.shape(self.Prec)[0] == self.rows
            and np.shape(self.ET)[0] == self.rows
            and np.shape(self.Temp)[0] == self.rows
        ), "all input data should have the same number of rows"
        assert (
            np.shape(self.Prec)[1] == self.cols
            and np.shape(self.ET)[1] == self.cols
            and np.shape(self.Temp)[1] == self.cols
        ), "all input data should have the same number of columns"
        assert (
            np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2]
        ), "all meteorological input data should have the same length"

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in Basic_inputs.keys() for i in range(4)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) is dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) is dict, "history_fname should be of type string "

        print("Calibration starts")

        # calculate the objective function
        def opt_fun(par):
            try:
                # distribute the parameters
                SpatialVarFun.Function(
                    par
                )  # , kub=SpatialVarFun.Kub, klb=SpatialVarFun.Klb, Maskingum=SpatialVarFun.Maskingum
                self.Parameters = SpatialVarFun.Par3d
                # run the model
                Wrapper.FW1(self)
                # calculate performance of the model
                try:
                    # error = self.objective_function(self.QGauges, self.qout, self.quz_routed, self.qlz_translated,*[self.GaugesTable])
                    error = self.objective_function(
                        self.QGauges, self.qout, *[self.GaugesTable]
                    )
                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(round(error, 3))
                    print(par)

                fail = 0
            except:
                error = np.nan
                fail = 1

            return error, [], fail

        # define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)
        for i in range(len(self.LB)):
            opt_prob.addVar(
                "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
            )

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res

    def lumpedCalibration(self, Basic_inputs, OptimizationArgs, printError=None):
        """runCalibration.

        this function runs the calibration algorithm for the Lumped conceptual hydrological model

        Parameters
        ----------
        ConceptualModel:
            [function] conceptual model and it should contain a function called simulate
        data:
            [numpy array] meteorological data as array with the first column as precipitation
            second as evapotranspiration, third as temperature and forth column as
            long term average temperature
        Basic_inputs:
            1-p2:
                [List] list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            2-init_st:
                [list] initial values for the state variables [sp,sm,uz,lz,wc] in mm
            3-UB:
                [Numeric] upper bound of the values of the parameters
            4-LB:
                [Numeric] Lower bound of the values of the parameters
        Q_obs:
            [Numeric] Observed values of discharge

        lumpedParNo:
            [int] nomber of lumped parameters, you have to enter the value of
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        lumpedParPos:
            [List] list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        objective_function:
            [function] objective function to calculate the performance of the model
            and to be used in the calibration
        *args:
            other arguments needed on the objective function

        Returns
        -------
        st: [4D array]
            state variables
        q_out: [1D array]
            calculated Discharge at the outlet of the catchment
        q_uz: [3D array]
            Distributed discharge for each cell

        Examples
        --------
            prec_path = prec_path="meteodata/4000/calib/prec"
            evap_path = evap_path="meteodata/4000/calib/evap"
            temp_path = temp_path="meteodata/4000/calib/temp"
            flow_acc_path = "GIS/4000/acc4000.tif"
            flow_direction_path = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/"+"parameters.txt"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(prec_path,evap_path,temp_path,DemPath,
                                              flow_acc_path,flow_direction_path,ParPath,p2)
        """
        # basic inputs
        # check if all inputs are included
        assert all(
            ["Route", "RoutingFn"][i] in Basic_inputs.keys() for i in range(2)
        ), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        Route = Basic_inputs["Route"]
        RoutingFn = Basic_inputs["RoutingFn"]
        if "InitialValues" in Basic_inputs.keys():
            InitialValues = Basic_inputs["InitialValues"]
        else:
            InitialValues = []

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert isinstance(ApiObjArgs, dict), "store_history should be 0 or 1"
        assert isinstance(ApiSolveArgs, dict), "history_fname should be of type string "

        # assert history_fname[-4:] == ".txt", "history_fname should be txt file please change extension or add .txt ad the end of the history_fname"

        print("Calibration starts")

        ### calculate the objective function
        def opt_fun(par):
            try:
                # parameters
                self.Parameters = par
                # run the model
                Wrapper.Lumped(self, Route, RoutingFn)
                # calculate performance of the model
                try:
                    error = self.objective_function(
                        self.QGauges[self.QGauges.columns[-1]], self.Qsim, *self.OFArgs
                    )
                    g = [
                        2 * par[-2] * par[-1] / self.dt,
                        (2 * par[-2] * (1 - par[-1])) / self.dt,
                    ]
                except TypeError:  # if no of inputs less than what the function needs
                    assert (
                        False
                    ), "the objective function you have entered needs more inputs please enter then in a list as *args"

                if printError != 0:
                    print(
                        f"Error = {round(error, 3)} Inequality Const = {np.round(g, 2)}"
                    )
                    # print(par)
                fail = 0
            except:
                error = np.nan
                g = []
                fail = 1
            return error, g, fail

        ### define the optimization components
        opt_prob = Optimization("HBV Calibration", opt_fun)

        if InitialValues != []:
            for i in range(len(self.LB)):
                opt_prob.addVar(
                    "x{0}".format(i),
                    type="c",
                    lower=self.LB[i],
                    upper=self.UB[i],
                    value=InitialValues[i],
                )
        else:
            for i in range(len(self.LB)):
                opt_prob.addVar(
                    "x{0}".format(i), type="c", lower=self.LB[i], upper=self.UB[i]
                )

        opt_prob.addObj("f")

        opt_prob.addCon("g1", "i")
        opt_prob.addCon("g2", "i")
        # print(opt_prob)
        opt_engine = HSapi(pll_type=pll_type, options=ApiObjArgs)

        # parse the ApiSolveArgs inputs
        # availablekeys = ['store_sol',"display_opts","store_hst","hot_start"]
        store_sol = ApiSolveArgs["store_sol"]
        display_opts = ApiSolveArgs["display_opts"]
        store_hst = ApiSolveArgs["store_hst"]
        hot_start = ApiSolveArgs["hot_start"]

        # for i in range(len(availablekeys)):
        # if availablekeys[i] in ApiSolveArgs.keys():
        # exec(availablekeys[i] + "=" + str(ApiSolveArgs[availablekeys[i]]))
        # print(availablekeys[i] + " = " + str(ApiSolveArgs[availablekeys[i]]))

        res = opt_engine(
            opt_prob,
            store_sol=store_sol,
            display_opts=display_opts,
            store_hst=store_hst,
            hot_start=hot_start,
        )

        self.OFvalue = res[0]
        self.Parameters = res[1]

        return res
