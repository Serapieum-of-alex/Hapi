"""Cross section model Created on Fri Apr  3 09:33:24 2020.

@author: mofarrag
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels import api as sm


class CrossSections:
    """Cross-Section class."""

    def __init__(self, name):
        self.name = name

    def reg_plot(
        self,
        x,
        y,
        xlab,
        ylab,
        xlgd,
        ylgd,
        title,
        filename,
        log,
        logandlinear,
        seelim,
        Save=False,
        *args,
        **kwargs
    ):
        """reg_plot.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        minmax_XS_area : TYPE
            DESCRIPTION.
        xlab : TYPE
            DESCRIPTION.
        ylab : TYPE
            DESCRIPTION.
        xlgd : TYPE
            DESCRIPTION.
        ylgd : TYPE
            DESCRIPTION.
        title : TYPE
            DESCRIPTION.
        filename : TYPE
            DESCRIPTION.
        log : [Bool]
            # Plot for log-log regression.
        logandlin : TYPE
            DESCRIPTION.
        seelim : [Bool]
            to draw vertical lines on the min and max drainage area on the graph.
        Save : TYPE, optional
            DESCRIPTION. The default is False.
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.


        Use the ordinary least squares to make a regression and plot the
        output.

        This function was developed first. The two following functions
        have been adapted from this one to meet some particular requirements.
        """
        for key in kwargs.keys():
            if key == "XLim":
                xmin = kwargs["XLim"][0]
                xmax = kwargs["XLim"][1]

        #    if log is True:
        #        # Linear regression with the ordinaty least squares method (Y, X).
        #        # sm.add_constant is required to get the intercept.
        #        # NaN values are dropped.
        #        results = sm.OLS(np.log10(y), sm.add_constant(np.log10(x)),
        #                         missing='drop').fit()
        #    elif log is False:
        #        results = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
        # fit the relation between log(x) and log(y)
        results_log = sm.OLS(
            np.log10(y), sm.add_constant(np.log10(x)), missing="drop"
        ).fit()
        # fit the relation between x and y
        results_lin = sm.OLS(y, sm.add_constant(x), missing="drop").fit()
        # Print the results in the console
        #    print(results.summary())
        print(results_log.summary())
        print(results_lin.summary())

        # Retrieve the intercept and the slope
        intercept = results_lin.params[0]
        slope = results_lin.params[1]

        # Transform to log-type
        #    coeff = 10**intercept
        #    exp = slope
        coeff = 10 ** results_log.params[0]
        exp = results_log.params[1]

        # Transform to log-type if required
        #    if log is True:
        #        coeff = 10**intercept
        #        exp = slope

        # Retrieve r2
        rsq_log = results_log.rsquared
        rsq_lin = results_lin.rsquared

        # Save them to a datafram using an external function
        #    results_df = results_summary_to_dataframe(results)
        #    results_df.index = ['intercept_' + filename, 'slope_' + filename]
        #    results_df.to_csv(filename + '.csv', sep=',')

        if logandlinear is False:
            # logarithmic data
            results_log_df = self.results_summary_to_dataframe(results_log)
            results_log_df.index = ["intercept_" + filename, "slope_" + filename]
            if Save:
                results_log_df.to_csv(filename + "_powerlaw.csv", sep=",")
            # linear data
            results_lin_df = self.results_summary_to_dataframe(results_lin)
            results_lin_df.index = ["intercept_" + filename, "slope_" + filename]
            if Save:
                results_lin_df.to_csv(filename + "_lin.csv", sep=",")

        if logandlinear is False:
            # Plot the points and the regression line
            plt.scatter(x, y)
            x_plot = np.linspace(x.min(), x.max(), 1000)

            # Plot for log-log regression
            if log is True:
                plt.plot(x_plot, coeff * (x_plot**exp))
                plt.xscale("log")
                plt.yscale("log")
                plt.annotate(
                    "$%s = %.4f%s^{%.4f}$" % (ylgd, coeff, xlgd, exp)
                    + "\n"
                    + "$R^2 = %.4f$" % rsq_log,
                    xy=(0.05, 0.90),
                    xycoords="axes fraction",
                )
                plt.xlim(0.5 * x.min(), 1.5 * x.max())
                plt.ylim(0.5 * y.min(), 1.5 * y.max())
            # Plot for linear regression
            elif log is False:
                plt.plot(x_plot, intercept + slope * x_plot)
                plt.annotate(
                    "%s = %.4f + %.4f%s" % (ylgd, intercept, slope, xlgd)
                    + "\n"
                    + "$R^2 = %.4f$" % rsq_lin,
                    xy=(0.05, 0.90),
                    xycoords="axes fraction",
                )
                plt.xlim(0, x.max() + 0.1 * (x.max() - x.min()))
                plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))

            plt.xlabel(xlab)
            plt.ylabel(ylab)
            # vertical lines for min and max cross section area
            if seelim is True:
                plt.axvline(x=xmin, color="red", ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color="red", ymin=0.25, ymax=0.75)
            plt.title(title)

            if Save:
                plt.savefig(filename + ".png", dpi=400)
                plt.close()
            if log is True:
                sm.graphics.plot_regress_exog(results_log, 1)
                if Save:
                    plt.savefig(filename + "_powerlaw_resid.png", dpi=400)
            elif log is False:
                sm.graphics.plot_regress_exog(results_lin, 1)
                if Save:
                    plt.savefig(filename + "_lin_resid.png", dpi=400)
                    plt.close()

        # Plot the power law and linear regressions on a log-log scale
        # and on a linear scale to compare the fits
        elif logandlinear is True:
            # Plot on a log-log scale
            plt.scatter(x, y, c="grey", marker="x")
            x_plot = np.linspace(x.min(), x.max(), 1000)
            plt.plot(x_plot, coeff * (x_plot**exp))
            plt.xscale("log")
            plt.yscale("log")
            plt.annotate(
                "$%s = %.4f%s^{%.4f}$" % (ylgd, coeff, xlgd, exp)
                + "\n"
                + "$R^2 = %.4f$" % rsq_log,
                xy=(0.05, 0.90),
                xycoords="axes fraction",
            )
            plt.plot(x_plot, intercept + slope * x_plot)
            plt.annotate(
                "$%s = %.4f + %.4f%s$" % (ylgd, intercept, slope, xlgd)
                + "\n"
                + "$R^2 = %.4f$" % rsq_lin,
                xy=(0.05, 0.80),
                xycoords="axes fraction",
            )
            plt.xlim(0.5 * x.min(), 1.5 * x.max())
            plt.ylim(0.5 * y.min(), 1.5 * y.max())
            plt.xlabel(xlab)
            plt.ylabel(ylab)

            if seelim is True:
                plt.axvline(x=xmin, color="red", ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color="red", ymin=0.25, ymax=0.75)
            plt.title(title)
            if Save:
                plt.savefig(filename + "_logscale.png", dpi=400)
                plt.close()
            # Plot on a linear scale
            plt.scatter(x, y, c="grey", marker="x")
            x_plot = np.linspace(x.min(), x.max(), 1000)
            plt.plot(x_plot, coeff * (x_plot**exp))
            plt.annotate(
                "$%s = %.4f%s^{%.4f}$" % (ylgd, coeff, xlgd, exp)
                + "\n"
                + "$R^2 = %.4f$" % rsq_log,
                xy=(0.05, 0.90),
                xycoords="axes fraction",
            )
            plt.plot(x_plot, intercept + slope * x_plot)
            plt.annotate(
                "$%s = %.4f + %.4f%s$" % (ylgd, intercept, slope, xlgd)
                + "\n"
                + "$R^2 = %.4f$" % rsq_lin,
                xy=(0.05, 0.80),
                xycoords="axes fraction",
            )
            plt.xlim(0, x.max() + 0.1 * (x.max() - x.min()))
            plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            if seelim is True:
                plt.axvline(x=xmin, color="red", ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color="red", ymin=0.25, ymax=0.75)
            plt.title(title)
            if Save:
                plt.savefig(filename + "_linscale.png", dpi=400)
                plt.close()

    def reg_plot_river(
        self, river_lst, data, minmax_XS_area, filename, log, Save=False, *args, **kargs
    ):
        """reg_plot_river.

        Use the ordinary least squares to make a regression and plot the
        output. This version makes use of the field 'river' in data to
        define a subset of gauges used for the regression.
        """
        # for key in kwargs.keys():
        #     if key == "XLim":
        #         xmin = kwargs['XLim'][0]
        #         xmax = kwargs['XLim'][1]
        #     if key == "area":
        #         minmax_XS_area = kwargs['area']

        df_output = pd.DataFrame()
        fig1 = plt.figure(figsize=(11.69, 16.53))
        for i, riveri in enumerate(river_lst):
            fig1.add_subplot(5, 2, i + 1)
            # The subset is defined checking the filed 'river' and using the
            # river names provided in river_lst
            datai = data[data["river"].str.contains(riveri, case=False) == True]
            # Warning, this df is passed outside of the function.
            min_XSUSarea = minmax_XS_area.at[riveri, "Min_UpXSArea"]
            max_XSUSarea = minmax_XS_area.at[riveri, "Max_UpXSArea"]
            print("There are {} gauges related to {}".format(datai.shape[0], riveri))
            # The regression is done if there are at least 4 gauges
            if datai.shape[0] > 3:
                x = datai["Area_Final"]
                y = datai["q2"]
                plt.scatter(x, y)
                X_plot = np.linspace(x.min(), x.max(), 1000)
                plt.ylabel("$HQ_2 [m^3/s]$")
                plt.xlabel("$Drainage Area [km^2]$")
                plt.axvline(x=min_XSUSarea, color="red", ymin=0.25, ymax=0.75)
                plt.axvline(x=max_XSUSarea, color="red", ymin=0.25, ymax=0.75)
                # Log-Log regression
                if log is True:
                    resultslog = sm.OLS(
                        np.log10(y),
                        sm.add_constant(np.log10(x)),  # Required to get the interc.
                        missing="drop",
                    ).fit()  # NaN values are dropped.
                    print(resultslog.summary())
                    results_df = self.results_summary_to_dataframe(resultslog)
                    results_df.index = ["interceptlog_" + riveri, "slopelog_" + riveri]
                    df_output = df_output.append(results_df)
                    coeffi = 10 ** resultslog.params[0]
                    expi = resultslog.params[1]
                    rsqi = resultslog.rsquared
                    plt.plot(
                        X_plot,
                        (10 ** resultslog.params[0]) * (X_plot ** resultslog.params[1]),
                    )
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.annotate(
                        "$HQ_2 = %.3fA^{%.3f}$" % (coeffi, expi)
                        + "\n"
                        + "$R^2 = %.3f$" % rsqi,
                        xy=(0.05, 0.85),
                        xycoords="axes fraction",
                    )
                    plt.xlim(0.5 * x.min(), 1.5 * x.max())
                    plt.ylim(0.5 * y.min(), 1.5 * y.max())
                    plt.title(filename + " " + riveri + " log-log")
                # Linear regression
                elif log is False:
                    results = sm.OLS(
                        y,
                        sm.add_constant(x),  # Required to get the interc.
                        missing="drop",
                    ).fit()  # NaN values are dropped.
                    print(results.summary())
                    results_df = self.results_summary_to_dataframe(results)
                    results_df.index = [
                        "interceptlinear_" + riveri,
                        "slopelinear_" + riveri,
                    ]
                    df_output = df_output.append(results_df)
                    intercept = results.params[0]
                    slope = results.params[1]
                    rsq = results.rsquared
                    plt.plot(X_plot, intercept + slope * X_plot)
                    plt.annotate(
                        "Q2 = %.4f + %.4fA" % (intercept, slope)
                        + "\n"
                        + "$R^2 = %.4f$" % rsq,
                        xy=(0.05, 0.85),
                        xycoords="axes fraction",
                    )
                    plt.xlim(0, 1.1 * x.max())
                    plt.ylim(0, 1.1 * y.max())
                    plt.title(filename + " " + riveri + " linear")
                # plt.show()
        plt.tight_layout()
        # Plot the residuals.
        # Export for log-log regression
        if log is True:
            if Save:
                fig1.savefig(filename + "_LogLog.png", dpi=400)
            fig2 = sm.graphics.plot_regress_exog(resultslog, 1)
            if Save:
                fig2.savefig(filename + "_ResidLogLog.png", dpi=400)
                df_output.to_csv(filename + "_log.csv", sep=",")
        # Export for linear regression
        elif log is False:
            if Save:
                fig1.savefig(filename + "_Linear.png", dpi=400)
            fig2 = sm.graphics.plot_regress_exog(results, 1)
            if Save:
                fig2.savefig(filename + "_ResidLinear.png", dpi=400)
                df_output.to_csv(filename + "_linear.csv", sep=",")
        if Save:
            plt.close(fig1)
            plt.close(fig2)

    def reg_plot_subbasin(
        self, subbasin_lst, data, minmax_XS_area, filename, log, redfact, Save=False
    ):
        """reg_plot_subbasin Use the ordinary least squares to make a regression and plot the output.

        This version makes use of the field 'Subbasin' in data to define a subset
        of gauges used for the regression.

        redfact is a real between 0 and 1, and is used to define for each
        subbasin an area threshold that will limit the number of small gauges
        selected.
        """
        df_output = pd.DataFrame()
        fig1 = plt.figure(figsize=(11.69, 16.53))
        for i, subbasini in enumerate(subbasin_lst):
            fig1.add_subplot(5, 2, i + 1)
            datai = data[data["Subbasin"].str.contains(subbasini, case=False) == True]
            min_XSUSarea = minmax_XS_area.at[subbasini, "Min_UpXSArea"]
            min_gauge_area = redfact * min_XSUSarea
            max_XSUSarea = minmax_XS_area.at[subbasini, "Max_UpXSArea"]
            datai_red = datai[datai["Area_Final"] > min_gauge_area]
            print(
                "There are {} gauges used for {}".format(datai_red.shape[0], subbasini)
            )
            if datai.shape[0] > 3:
                x = datai_red["Area_Final"]
                y = datai_red["q2"]
                plt.scatter(x, y)
                X_plot = np.linspace(x.min(), x.max(), 1000)

                if log is True:
                    resultslog = sm.OLS(
                        np.log10(y),
                        sm.add_constant(np.log10(x)),  # Required to get the interc.
                        missing="drop",
                    ).fit()  # NaN values are dropped.
                    print(resultslog.summary())
                    results_df = self.results_summary_to_dataframe(resultslog)
                    results_df.index = [
                        "interceptlog_" + subbasini,
                        "slopelog_" + subbasini,
                    ]
                    df_output = df_output.append(results_df)
                    coeff = 10 ** resultslog.params[0]
                    exp = resultslog.params[1]
                    rsq = resultslog.rsquared
                    plt.plot(
                        X_plot,
                        (10 ** resultslog.params[0]) * (X_plot ** resultslog.params[1]),
                    )
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.annotate(
                        "$Q2 = %.3fA^{%.3f}$" % (coeff, exp)
                        + "\n"
                        + "$R^2 = %.3f$" % rsq,
                        xy=(0.05, 0.85),
                        xycoords="axes fraction",
                    )
                    # plt.xlim(0.5*x.min(), 1.5*x.max())
                    plt.xlim(0.5 * min_gauge_area, 1.5 * max_XSUSarea)
                    plt.ylim(0.5 * y.min(), 1.5 * y.max())
                    plt.title(filename + " " + subbasini + " log-log")
                elif log is False:
                    results = sm.OLS(
                        y,
                        sm.add_constant(x),  # Required to get the interc.
                        missing="drop",
                    ).fit()  # NaN values are dropped.
                    print(results.summary())
                    results_df = self.results_summary_to_dataframe(results)
                    results_df.index = [
                        "interceptlinear_" + subbasini,
                        "slopelinear_" + subbasini,
                    ]
                    df_output = df_output.append(results_df)
                    intercept = results.params[0]
                    slope = results.params[1]
                    rsq = results.rsquared
                    plt.plot(X_plot, intercept + slope * X_plot)
                    plt.annotate(
                        "Q2 = %.4f + %.4fA" % (intercept, slope)
                        + "\n"
                        + "$R^2 = %.4f$" % rsq,
                        xy=(0.05, 0.85),
                        xycoords="axes fraction",
                    )
                    # plt.xlim(0.8*x.min(), 1.1*x.max())
                    plt.xlim(0, 1.1 * max_XSUSarea)
                    plt.ylim(0, 1.1 * y.max())
                    plt.title(filename + " " + subbasini + " linear")

                plt.ylabel("Q2 [m3/s]")
                plt.xlabel("Drainage Area [km2]")
                plt.axvline(x=min_XSUSarea, color="red", ymin=0.25, ymax=0.75)
                plt.axvline(x=max_XSUSarea, color="red", ymin=0.25, ymax=0.75)
                # plt.show()
        plt.tight_layout()
        if log is True:
            if Save:
                fig1.savefig(filename + "_LogLog.png", dpi=400)
                plt.close(fig1)
            fig2 = sm.graphics.plot_regress_exog(resultslog, 1)
            if Save:
                fig2.savefig(filename + "_ResidLogLog.png", dpi=400)
                df_output.to_csv(filename + "_LogLog.csv", sep=",")
        elif log is False:
            if Save:
                fig1.savefig(filename + "_Linear.png", dpi=400)
                plt.close(fig1)
            fig2 = sm.graphics.plot_regress_exog(results, 1)
            if Save:
                fig2.savefig(filename + "_ResidLinear.png", dpi=400)
                plt.close(fig2)
                df_output.to_csv(filename + "_Linear.csv", sep=",")

    @staticmethod
    def results_summary_to_dataframe(res):
        """Take the result of a statsmodel results table and transforms it into a dataframe."""

        pvals = res.pvalues
        coeff = res.params
        conf_lower = res.conf_int()[0]
        conf_higher = res.conf_int()[1]
        stderr = res.bse
        tvals = res.tvalues
        rsq = res.rsquared
        rsq_adj = res.rsquared_adj
        no_obs = res.nobs

        res_df = pd.DataFrame(
            {
                "pvals": pvals,
                "coeff": coeff,
                "conf_lower": conf_lower,
                "conf_higher": conf_higher,
                "std_err": stderr,
                "tvals": tvals,
                "R-squared": rsq,
                "R-squared_adj": rsq_adj,
                "no_obs": no_obs,
            }
        )

        # Reordering...
        res_df = res_df[
            [
                "coeff",
                "std_err",
                "tvals",
                "pvals",
                "conf_lower",
                "conf_higher",
                "R-squared",
                "R-squared_adj",
                "no_obs",
            ]
        ]
        return res_df

    def ListAttributes(self):
        """Print Attributes List."""

        print("\n")
        print(
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")
