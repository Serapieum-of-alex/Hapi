from collections import OrderedDict
from typing import Union, Any, Tuple, List #Dict,
# import types

import matplotlib.pyplot as plt
import numpy as np

# import scipy as sp
import scipy.optimize as so
from loguru import logger
from matplotlib import gridspec
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import randint
from scipy.stats import chisquare, genextreme, gumbel_r, ks_2samp, norm

from Hapi.sm.parameterestimation import Lmoments
from Hapi.sm.statisticaltools import StatisticalTools as st

ninf = 1e-5


class PlottingPosition:
    def __init__(self):
        pass

    @staticmethod
    def Weibul(data: Union[list, np.ndarray], option: int = 1) -> np.ndarray:
        """Weibul.

        Weibul method to calculate the cumulative distribution function CDF or
        return period.

        Parameters
        ----------
        data : [list/array]
            list/array of the data.
        option : [1/2]
            1 to calculate the cumulative distribution function cdf or
            2 to calculate the return period.default=1

        Returns
        -------
        CDF/T: [list]
            list of cumulative distribution function or return period.
        """
        data.sort()

        if option == 1:
            CDF = np.array([j / (len(data) + 1) for j in range(1, len(data) + 1)])
            return CDF
        else:
            CDF = [j / (len(data) + 1) for j in range(1, len(data) + 1)]
            T = np.array([1 / (1 - j) for j in CDF])
            return T

    @staticmethod
    def Returnperiod(F: Union[list, np.ndarray]) -> np.ndarray:
        F = np.array(F)
        T = 1 / (1 - F)
        return T


class Gumbel:
    def __init__(
            self,
            data: Union[list, np.ndarray]=None,
            loc: Union[int, float]=None,
            scale: Union[int, float]=None
    ):
        """
        data : [list]
            data time series.
        """
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = np.array(data)
            self.data_sorted = np.sort(data)
            self.cdf_Weibul = PlottingPosition.Weibul(data)
            self.KStable = 1.22 / np.sqrt(len(self.data))

        self.loc = loc
        self.scale = scale
        self.Dstatic = None
        self.KS_Pvalue = None
        self.chistatic = None
        self.chi_Pvalue = None

        pass

    def pdf(
        self,
        loc: Union[float, int],
        scale: Union[float, int],
        plot_figure: bool=False,
        figsize: tuple=(6, 5),
        xlabel: str="Actual data",
        ylabel: str="pdf",
        fontsize: Union[float, int]=15,
        actualdata: Union[bool, np.ndarray]=True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x .

        Parameters:
        -----------
        loc : [numeric]
            location parameter of the gumbel distribution.
        scale : [numeric]
            scale parameter of the gumbel distribution.

        Returns
        -------
        pdf : [array]
            probability density function pdf.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        if isinstance(actualdata, bool):
            ts = self.data
        else:
            ts = actualdata

        z = (ts - loc) / scale
        pdf = (1.0 / scale) * (np.exp(-(z + (np.exp(-z)))))
        # gumbel_r.pdf(data, loc=loc, scale=scale)
        if plot_figure:
            Qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            pdf_fitted = self.pdf(loc, scale, actualdata=Qx)

            fig, ax = plot.pdf(
                Qx,
                pdf_fitted,
                self.data_sorted,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )
            return pdf, fig, ax
        else:
            return pdf

    def cdf(
        self,
        loc: Union[float, int],
        scale: Union[float, int],
        plot_figure: bool=False,
        figsize: tuple=(6, 5),
        xlabel: str="data",
        ylabel: str="cdf",
        fontsize: int=15,
        actualdata: Union[bool, np.ndarray]=True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
            1- loc : [numeric]
                location parameter of the gumbel distribution.
            2- scale : [numeric]
                scale parameter of the gumbel distribution.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        if isinstance(actualdata, bool):
            ts = self.data
        else:
            ts = actualdata

        z = (ts - loc) / scale
        cdf = np.exp(-np.exp(-z))

        if plot_figure:
            Qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            cdf_fitted = self.cdf(loc, scale, actualdata=Qx)

            cdf_Weibul = PlottingPosition.Weibul(self.data_sorted)

            fig, ax = plot.cdf(
                Qx,
                cdf_fitted,
                self.data_sorted,
                cdf_Weibul,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )

            return cdf, fig, ax
        else:
            return cdf
        # gumbel_r.cdf(Q, loc, scale)

    @staticmethod
    def ObjectiveFn(p, x):
        """ObjectiveFn.

        Link : https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize
        :param p:
        :param x:
        :return:
        """
        threshold = p[0]
        loc = p[1]
        scale = p[2]

        x1 = x[x < threshold]
        nx2 = len(x[x >= threshold])
        # pdf with a scaled pdf
        # L1 is pdf based
        L1 = (-np.log((Gumbel.pdf(x1, loc, scale) / scale))).sum()
        # L2 is cdf based
        L2 = (-np.log(1 - Gumbel.cdf(threshold, loc, scale))) * nx2
        # print x1, nx2, L1, L2
        return L1 + L2

    def EstimateParameter(self, method: str="mle", ObjFunc=None,
                          threshold:Union[None, float, int]=None, Test: bool=True) -> tuple:
        """EstimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        ObjFunc : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        Test: [bool]
            Default is True.

        Returns
        -------
        Param : [list]
            scale and location parameter of the gumbel distribution.
            [loc, scale]
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = method.lower()
        if not method in ["mle", "mm", "lmoments", "optimization"]:
            raise ValueError(
                method + "value should be 'mle', 'mm', 'lmoments' or 'optimization'"
            )
        if method == "mle" or method == "mm":
            Param = list(gumbel_r.fit(self.data, method=method))
        elif method == "lmoments":
            LM = Lmoments(self.data)
            LMU = LM.Lmom()
            Param = Lmoments.Gumbel(LMU)
        elif method == "optimization":
            if ObjFunc is None or threshold is None:
                raise TypeError("threshold should be numeric value")
            Param = gumbel_r.fit(self.data, method=method)
            # then we use the result as starting value for your truncated Gumbel fit
            Param = so.fmin(
                ObjFunc,
                [threshold, Param[0], Param[1]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            Param = [Param[1], Param[2]]

        self.loc = Param[0]
        self.scale = Param[1]

        if Test:
            self.ks()
            self.chisquare()

        return Param


    @staticmethod
    def TheporeticalEstimate(
            loc: Union[float, int],
            scale: Union[float, int],
            cdf: np.ndarray
) -> np.ndarray:
        """TheporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution

        Parameters:
        -----------
            1- param : [list]
                location ans scale parameters of the gumbel distribution.
            2- cdf: [list]
                cummulative distribution function/ Non Exceedence probability.
        Return:
        -------
            1- theoreticalvalue : [numeric]
                Value based on the theoretical distribution
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        if any(cdf) <= 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        cdf = np.array(cdf)
        Qth = loc - scale * (np.log(-np.log(cdf)))

        # the main equation form scipy
        # Qth = gumbel_r.ppf(F, loc=param_dist[0], scale=param_dist[1])
        return Qth


    def ks(self) -> tuple:
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Value of loc/scale parameter is unknown please use "
                "'EstimateParameter' to obtain them"
            )
        Qth = self.TheporeticalEstimate(self.loc, self.scale, self.cdf_Weibul)

        test = ks_2samp(self.data, Qth)
        self.Dstatic = test.statistic
        self.KS_Pvalue = test.pvalue

        print("-----KS Test--------")
        print(f"Statistic = {test.statistic}")
        if self.Dstatic < self.KStable:
            print("Accept Hypothesis")
        else:
            print("reject Hypothesis")
        print(f"P value = {test.pvalue}")
        return test.statistic, test.pvalue


    def chisquare(self) -> tuple:
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Value of loc/scale parameter is unknown please use "
                "'EstimateParameter' to obtain them"
            )

        Qth = self.TheporeticalEstimate(self.loc, self.scale, self.cdf_Weibul)
        try:
            test = chisquare(st.Standardize(Qth), st.Standardize(self.data))
            self.chistatic = test.statistic
            self.chi_Pvalue = test.pvalue
            print("-----chisquare Test-----")
            print("Statistic = " + str(test.statistic))
            print("P value = " + str(test.pvalue))
            return test.statistic, test.pvalue
        except Exception as e:
            print(e)
            # raise


    def ConfidenceInterval(
            self,
            loc: Union[float, int],
            scale: Union[float, int],
            F: np.ndarray,
            alpha: float=0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ConfidenceInterval.

        Parameters:
        -----------
            1- loc : [numeric]
                location parameter of the gumbel distribution.
            2- scale : [numeric]
                scale parameter of the gumbel distribution.
            1-F : [list]
                Non Exceedence probability
            3-alpha : [numeric]
                alpha or SignificanceLevel is a value of the confidence interval.

        Return:
        -------
            Qupper : [list]
                upper bound coresponding to the confidence interval.
            Qlower : [list]
                lower bound coresponding to the confidence interval.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        Qth = self.TheporeticalEstimate(loc, scale, F)
        Y = [-np.log(-np.log(j)) for j in F]
        StdError = [
            (scale / np.sqrt(len(self.data)))
            * np.sqrt(1.1087 + 0.5140 * j + 0.6079 * j ** 2)
            for j in Y
        ]
        v = norm.ppf(1 - alpha / 2)
        Qupper = np.array([Qth[j] + v * StdError[j] for j in range(len(self.data))])
        Qlower = np.array([Qth[j] - v * StdError[j] for j in range(len(self.data))])
        return Qupper, Qlower

    def ProbapilityPlot(
        self,
        loc: float,
        scale: float,
        F: np.ndarray,
        alpha: float=0.1,
        fig1size: tuple=(10, 5),
        fig2size: tuple=(6, 6),
        xlabel: str="Actual data",
        ylabel: str="cdf",
        fontsize: int=15,
    ) -> Tuple[List[Figure], list]:
        """ProbapilityPlot.

        ProbapilityPlot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
            loc : [numeric]
                location parameter of the gumbel distribution.
            scale : [numeric]
                scale parameter of the gumbel distribution.
            F : [np.ndarray]
                theoretical cdf calculated using weibul or using the distribution cdf function.
            alpha : [float]
                value between 0 and 1.
            fig1size: [tuple]
                Default is (10, 5)
            fig2size: [tuple]
                Default is (6, 6)
            xlabel: [str]
                Default is "Actual data"
            ylabel: [str]
                Default is "cdf"
            fontsize: [float]
                Default is 15.

        Returns
        -------
        Qth : [list]
            theoretical generated values based on the theoretical cdf calculated from
            weibul or the distribution parameters.
        Qupper : [list]
            upper bound coresponding to the confidence interval.
        Qlower : [list]
            lower bound coresponding to the confidence interval.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        Qth = self.TheporeticalEstimate(loc, scale, F)
        Qupper, Qlower = self.ConfidenceInterval(loc, scale, F, alpha)

        Qx = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted = self.pdf(loc, scale, actualdata=Qx)
        cdf_fitted = self.cdf(loc, scale, actualdata=Qx)

        fig, ax = plot.details(
            Qx,
            Qth,
            self.data,
            pdf_fitted,
            cdf_fitted,
            F,
            Qlower,
            Qupper,
            alpha,
            fig1size=fig1size,
            fig2size=fig2size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax


class GEV:

    data: ndarray

    def __init__(self, data: Union[list, np.ndarray]=None, shape: Union[int, float]=None,
                 loc: Union[int, float]=None, scale: Union[int, float]=None):
        """
        data : [list]
            data time series.
        """
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = np.array(data)
            self.data_sorted = np.sort(data)
            self.cdf_Weibul = PlottingPosition.Weibul(data)
            self.KStable = 1.22 / np.sqrt(len(self.data))

        self.shape = shape
        self.loc = loc
        self.scale = scale
        self.Dstatic = None
        self.KS_Pvalue = None

        self.chistatic = None
        self.chi_Pvalue = None
        pass

    def pdf(
        self,
        shape: Union[float, int],
        loc: Union[float, int],
        scale: Union[float, int],
        plot_figure: bool=False,
        figsize: tuple=(6, 5),
        xlabel: str="Actual data",
        ylabel: str="pdf",
        fontsize: int=15,
        actualdata:Union[bool, np.ndarray]=True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of GEV's pdf with parameters loc and scale at x .

        Parameters
        ----------
        shape : [numeric]
            shape parameter.
        loc : [numeric]
            location parameter.
        scale : [numeric]
            scale parameter.
        plot_figure: [bool]
            Default is False.
        figsize: [tuple]
            Default is (6, 5).
        xlabel: [str]
            Default is "Actual data".
        ylabel: [str]
            Default is "pdf".
        fontsize: [int]
            Default is 15.
        actualdata : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if isinstance(actualdata, bool):
            ts = self.data_sorted
        else:
            ts = actualdata

        pdf = []
        for i in range(len(ts)):
            z = (ts[i] - loc) / scale
            if shape == 0:
                val = np.exp(-(z + np.exp(-z)))
                pdf.append((1 / scale) * val)
                continue

            y = 1 - shape * z
            if y > ninf:
                # np.log(y) = ln(y)
                # ln is the inverse of e
                lnY = (-1 / shape) * np.log(y)
                val = np.exp(-(1 - shape) * lnY - np.exp(-lnY))
                pdf.append((1 / scale) * val)
                continue

            # y = 1 + shape * z
            # if y > ninf:
            #     # np.log(y) = ln(y)
            #     # ln is the inverse of e
            #     Q = y ** (-1 / shape)
            #     val = np.power(Q,1 + shape) * np.exp(-Q)
            #     pdf.append((1 / scale) * val)
            #     continue

            if shape < 0:
                pdf.append(0)
                continue
            pdf.append(0)
            # pdf.append(1)

        if len(pdf) == 1:
            pdf = pdf[0]

        pdf = np.array(pdf)
        # genextreme.pdf(data, loc=loc, scale=scale, c=shape)
        if plot_figure:
            Qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            pdf_fitted = self.pdf(shape, loc, scale, actualdata=Qx)

            fig, ax = plot.pdf(
                Qx,
                pdf_fitted,
                self.data_sorted,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )
            return pdf, fig, ax
        else:
            return pdf

    def cdf(
        self,
        shape: Union[float, int],
        loc: Union[float, int],
        scale: Union[float, int],
        plot_figure: bool=False,
        figsize: tuple=(6, 5),
        xlabel: str="Actual data",
        ylabel: str="cdf",
        fontsize: int=15,
        actualdata: Union[bool, np.ndarray]=True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """cdf.

        Returns the value of Gumbel's cdf with parameters loc and scale at x.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        if isinstance(actualdata, bool):
            ts = self.data
        else:
            ts = actualdata

        z = (ts - loc) / scale
        if shape == 0:
            # GEV is Gumbel distribution
            cdf = np.exp(-np.exp(-z))
        else:
            y = 1 - shape * z
            cdf = list()
            for i in range(0, len(y)):
                if y[i] > ninf:
                    logY = -np.log(y[i]) / shape
                    cdf.append(np.exp(-np.exp(-logY)))
                elif shape < 0:
                    cdf.append(0)
                else:
                    cdf.append(1)

        cdf = np.array(cdf)

        if plot_figure:
            Qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            cdf_fitted = self.cdf(shape, loc, scale, actualdata=Qx)

            cdf_Weibul = PlottingPosition.Weibul(self.data_sorted)

            fig, ax = plot.cdf(
                Qx,
                cdf_fitted,
                self.data_sorted,
                cdf_Weibul,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )

            return cdf, fig, ax
        else:
            return cdf
        # genextreme.cdf()

    def EstimateParameter(self, method: str="mle", ObjFunc=None,
                          threshold: Union[int, float, None]=None,
                          Test: bool=True) -> tuple:
        """EstimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        ObjFunc : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        Test: bool
            Default is True

        Returns
        -------
        Param : [list]
            shape, loc, scale parameter of the gumbel distribution in that order.

        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = method.lower()
        if not method in ["mle", "mm", "lmoments", "optimization"]:
            raise ValueError(
                method + "value should be 'mle', 'mm', 'lmoments' or 'optimization'"
            )

        if method == "mle" or method == "mm":
            Param = list(genextreme.fit(self.data, method=method))
        elif method == "lmoments":
            LM = Lmoments(self.data)
            LMU = LM.Lmom()
            Param = Lmoments.GEV(LMU)
        elif method == "optimization":
            if ObjFunc is None or threshold is None:
                raise TypeError("ObjFunc and threshold should be numeric value")

            Param = genextreme.fit(self.data, method=method)
            # then we use the result as starting value for your truncated Gumbel fit
            Param = so.fmin(
                ObjFunc,
                [threshold, Param[0], Param[1], Param[2]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            Param = [Param[1], Param[2], Param[3]]

        self.shape = Param[0]
        self.loc = Param[1]
        self.scale = Param[2]

        if Test:
            self.ks()
            try:
                self.chisquare()
            except ValueError:
                print("chisquare test failed")

        return Param

    @staticmethod
    def TheporeticalEstimate(
            shape: Union[float, int],
            loc: Union[float, int],
            scale: Union[float, int],
            F: np.ndarray
    ) -> np.ndarray:
        """TheporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution

        Parameters:
        -----------
            1- param : [list]
                location ans scale parameters of the gumbel distribution.
            2- F : [list]
                cummulative distribution function/ Non Exceedence probability.
        Return:
        -------
            1- theoreticalvalue : [numeric]
                Value based on the theoretical distribution
        """
        if scale <= 0:
            raise ValueError("Parameters Invalid")

        if any(F) < 0 or any(F) > 1:
            raise ValueError("cdf Value Invalid")

        Qth = list()
        for i in range(len(F)):
            if F[i] <= 0 or F[i] >= 1:
                if F[i] == 0 and shape < 0:
                    Qth.append(loc + scale / shape)
                elif F[i] == 1 and shape > 0:
                    Qth.append(loc + scale / shape)
                else:
                    raise ValueError(str(F[i]) + " value of cdf is Invalid")
            # F = np.array(F)
            Y = -np.log(-np.log(F[i]))
            if shape != 0:
                Y = (1 - np.exp(-1 * shape * Y)) / shape

            Qth.append(loc + scale * Y)
        Qth = np.array(Qth)
        # the main equation from scipy
        # Qth = genextreme.ppf(F, shape, loc=loc, scale=scale)
        return Qth


    def ks(self):
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        if not hasattr(self, "loc") or not hasattr(self, "scale"):
            raise ValueError(
                "Value of loc/scale parameter is unknown please use "
                "'EstimateParameter' to obtain them"
            )
        Qth = self.TheporeticalEstimate(
            self.shape, self.loc, self.scale, self.cdf_Weibul
        )

        test = ks_2samp(self.data, Qth)
        self.Dstatic = test.statistic
        self.KS_Pvalue = test.pvalue
        print("-----KS Test--------")
        print("Statistic = " + str(test.statistic))
        if self.Dstatic < self.KStable:
            print("Accept Hypothesis")
        else:
            print("reject Hypothesis")
        print("P value = " + str(test.pvalue))

        return test.statistic, test.pvalue

    def chisquare(self):
        if not hasattr(self, "loc") or not hasattr(self, "scale"):
            raise ValueError(
                "Value of loc/scale parameter is unknown please use "
                "'EstimateParameter' to obtain them"
            )

        Qth = self.TheporeticalEstimate(
            self.shape, self.loc, self.scale, self.cdf_Weibul
        )

        test = chisquare(st.Standardize(Qth), st.Standardize(self.data))
        self.chistatic = test.statistic
        self.chi_Pvalue = test.pvalue
        print("-----chisquare Test-----")
        print("Statistic = " + str(test.statistic))
        print("P value = " + str(test.pvalue))

        return test.statistic, test.pvalue

    def ConfidenceInterval(
        self,
        shape: Union[float, int],
        loc: Union[float, int],
        scale: Union[float, int],
        F: np.ndarray,
        alpha: float=0.1,
        statfunction=np.average,
        n_samples: int=100,
        **kargs
    ):
        """ConfidenceInterval.

        Parameters:
        -----------
            1- loc : [numeric]
                location parameter of the gumbel distribution.
            2- scale : [numeric]
                scale parameter of the gumbel distribution.
            1-F : [list]
                Non Exceedence probability
            3-alpha : [numeric]
                alpha or SignificanceLevel is a value of the confidence interval.

        Return:
        -------
            Qupper : [list]
                upper bound coresponding to the confidence interval.
            Qlower : [list]
                lower bound coresponding to the confidence interval.
        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        Param = [shape, loc, scale]
        CI = ConfidenceInterval.BootStrap(
            self.data,
            statfunction=statfunction,
            gevfit=Param,
            F=F,
            alpha=alpha,
            n_samples=n_samples,
            **kargs
        )
        Qlower = CI["LB"]
        Qupper = CI["UB"]

        return Qupper, Qlower


    def ProbapilityPlot(
        self,
        shape: Union[float, int],
        loc: Union[float, int],
        scale: Union[float, int],
        F,
        alpha=0.1,
        func=None,
        n_samples=100,
        fig1size=(10, 5),
        fig2size=(6, 6),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=15,
    ):
        """ProbapilityPlot.

        ProbapilityPlot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        loc : [numeric]
            location parameter of the GEV distribution.
        scale : [numeric]
            scale parameter of the GEV distribution.
        shape: [float, int]
            shape parameter for the GEV distribution
        F : [list]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        alpha : [float]
            value between 0 and 1.
        fontsize : [numeric]
            font size of the axis labels and legend
        ylabel : [string]
            y label string
        xlabel : [string]
            x label string
        fig1size : [tuple]
            size of the pdf and cdf figure
        fig2size : [tuple]
            size of the confidence interval figure
        n_samples : [integer]
            number of points in the condidence interval calculation
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.
        func : [function]
            function to be used in the confidence interval calculation.

        """
        if scale <= 0:
            raise ValueError("Scale Parameter is negative")

        Qth = self.TheporeticalEstimate(shape, loc, scale, F)
        if func is None:
            func = ConfidenceInterval.GEVfunc

        Param_dist = [shape, loc, scale]
        CI = ConfidenceInterval.BootStrap(
            self.data, statfunction=func, gevfit=Param_dist, n_samples=n_samples, F=F
        )
        Qlower = CI["LB"]
        Qupper = CI["UB"]

        Qx = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted = self.pdf(shape, loc, scale, actualdata=Qx)
        cdf_fitted = self.cdf(shape, loc, scale, actualdata=Qx)

        fig, ax = plot.details(
            Qx,
            Qth,
            self.data,
            pdf_fitted,
            cdf_fitted,
            F,
            Qlower,
            Qupper,
            alpha,
            fig1size=fig1size,
            fig2size=fig2size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax


class ConfidenceInterval:
    def __init__():
        pass

    @staticmethod
    def BSIndexes(data, n_samples=10000):
        """
        Given data points data, where axis 0 is considered to delineate points, return
        an generator for sets of bootstrap indexes. This can be used as a list
        of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
        """
        for _ in range(n_samples):
            yield randint(data.shape[0], size=(data.shape[0],))


    def BootStrap(
            data: Union[list, np.ndarray],
            statfunction,
            alpha: float=0.05,
            n_samples: int=100,
            **kargs
    ): # ->  Dict[str, OrderedDict[str, Tuple[Any, Any]]]
        """
        Calculate confidence intervals using parametric bootstrap and the
        percentil interval method
        This is used to obtain confidence intervals for the estimators and
        the return values for several return values.

        More info about bootstrapping can be found on:
            - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
            - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

        parameters:
        -----------
        3-alpha : [numeric]
                alpha or SignificanceLevel is a value of the confidence interval.
        kwargs :
            gevfit : [list]
                list of the three parameters of the GEV distribution [shape, loc, scale]
            F : [list]
                non exceedence probability/ cdf
        """
        alphas = np.array([alpha / 2, 1 - alpha / 2])
        tdata = (np.array(data),)

        # We don't need to generate actual samples; that would take more memory.
        # Instead, we can generate just the indexes, and then apply the statfun
        # to those indexes.
        bootindexes = ConfidenceInterval.BSIndexes(tdata[0], n_samples)
        stat = np.array(
            [
                statfunction(*(x[indexes] for x in tdata), **kargs)
                for indexes in bootindexes
            ]
        )
        stat.sort(axis=0)

        # Percentile Interval Method
        avals = alphas
        nvals = np.round((n_samples - 1) * avals).astype("int")

        if np.any(nvals == 0) or np.any(nvals == n_samples - 1):
            logger.debug("Some values used extremal samples; results are probably unstable.")
            # warnings.warn(
            #     "Some values used extremal samples; results are probably unstable.",
            #     InstabilityWarning,
            # )
        elif np.any(nvals < 10) or np.any(nvals >= n_samples - 10):
            logger.debug("Some values used top 10 low/high samples; results may be unstable.")
            # warnings.warn(
            #     "Some values used top 10 low/high samples; results may be unstable.",
            #     InstabilityWarning,
            # )

        if nvals.ndim == 1:
            # All nvals are the same. Simple broadcasting
            out = stat[nvals]
        else:
            # Nvals are different for each data point. Not simple broadcasting.
            # Each set of nvals along axis 0 corresponds to the data at the same
            # point in other axes.
            out = stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]

        UB = out[0, 3:]
        LB = out[1, 3:]
        params = OrderedDict()
        params["shape"] = (out[0, 0], out[1, 0])
        params["location"] = (out[0, 1], out[1, 1])
        params["scale"] = (out[0, 2], out[1, 3])

        return {"LB": LB, "UB": UB, "params": params}

    # The function to bootstrap
    @staticmethod
    def GEVfunc(data, **kwargs):

        gevfit = kwargs["gevfit"]
        F = kwargs["F"]
        shape = gevfit[0]
        loc = gevfit[1]
        scale = gevfit[2]
        # generate theoretical estimates based on a random cdf, and the dist parameters
        sample = GEV.TheporeticalEstimate(shape, loc, scale, np.random.rand(len(data)))
        # get parameters based on the new generated sample
        LM = Lmoments(sample)
        mum = LM.Lmom()
        newfit = LM.GEV(mum)
        shape = newfit[0]
        loc = newfit[1]
        scale = newfit[2]
        # return period
        # T = np.arange(0.1, 999.1, 0.1) + 1
        # +1 in order not to make 1- 1/0.1 = -9
        # T = np.linspace(0.1, 999, len(data)) + 1
        # coresponding theoretical estimate to T
        # F = 1 - 1 / T
        Qth = GEV.TheporeticalEstimate(shape, loc, scale, F)

        res = newfit
        res.extend(Qth)
        return tuple(res)


class plot:
    def __init__(self):
        pass

    def pdf(
        Qx: np.ndarray,
        pdf_fitted,
        data_sorted: np.ndarray,
        figsize: tuple=(6, 5),
        xlabel: str="Actual data",
        ylabel: str="pdf",
        fontsize: int=15,
    ):

        fig = plt.figure(figsize=figsize)
        # gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax = fig.add_subplot()
        ax.plot(Qx, pdf_fitted, "r-")
        ax.hist(data_sorted, density=True)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        return fig, ax

    @staticmethod
    def cdf(
        Qx,
        cdf_fitted,
        data_sorted,
        cdf_Weibul,
        figsize=(6, 5),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=15,
    ):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.plot(Qx, cdf_fitted, "r-", label="Fitted distribution")
        ax.plot(data_sorted, cdf_Weibul, ".-", label="Weibul plotting position")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize, framealpha=1)
        return fig, ax

    @staticmethod
    def details(
        Qx: Union[np.ndarray, list],
        Qth: Union[np.ndarray, list],
        Qact: Union[np.ndarray, list],
        pdf: Union[np.ndarray, list],
        cdf_fitted: Union[np.ndarray, list],
        F: Union[np.ndarray, list],
        Qlower: Union[np.ndarray, list],
        Qupper: Union[np.ndarray, list],
        alpha: float,
        fig1size: tuple=(10, 5),
        fig2size: tuple=(6, 6),
        xlabel: str="Actual data",
        ylabel: str="cdf",
        fontsize: int=15,
    ) -> Tuple[List[Figure], List[Any]]:

        fig1 = plt.figure(figsize=fig1size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig1)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(Qx, pdf, "r-")
        ax1.hist(Qact, density=True)
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig1.add_subplot(gs[0, 1])
        ax2.plot(Qx, cdf_fitted, "r-")
        Qact.sort()
        ax2.plot(Qact, F, ".-")
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=15)

        fig2 = plt.figure(figsize=fig2size)
        plt.plot(Qth, Qact, "d", color="#606060", markersize=12, label="Actual Data")
        plt.plot(Qth, Qth, "^-.", color="#3D59AB", label="Theoretical Data")

        plt.plot(
            Qth,
            Qlower,
            "*--",
            color="#DC143C",
            markersize=12,
            label="Lower limit (" + str(int((1 - alpha) * 100)) + " % CI)",
        )
        plt.plot(
            Qth,
            Qupper,
            "*--",
            color="#DC143C",
            markersize=12,
            label="Upper limit (" + str(int((1 - alpha) * 100)) + " % CI)",
        )
        plt.legend(fontsize=fontsize, framealpha=1)
        plt.xlabel("Theoretical Values", fontsize=fontsize)
        plt.ylabel("Actual Values", fontsize=fontsize)

        return [fig1, fig2], [ax1, ax2]
