"""
Created on Mon Mar 29 21:32:29 2021

@author: mofarrag
"""
import matplotlib.pyplot as plt
import numpy as np

import Hapi.visualizer as V

Vis = V.Visualize(1)


class SensitivityAnalysis:
    """
    ==============================
        SensitivityAnalysis
    ==============================
    SensitivityAnalysis class

    Methods
        1- OAT
        2- Sobol

    """

    def __init__(self, Parameter, LB, UB, Function, Positions=[], NoValues=5, Type=1):
        """
        =============================================================================
            SensitivityAnalysis(self, Parameter, LB, UB, Function, Positions=[], NoValues=5, Type=1)
        =============================================================================
        To instantiate the SensitivityAnalysis class you have to provide the
        following parameters

        Parameters
        ----------
        Parameter : [dataframe]
            dataframe with the index as the name of the parameters and one column
            with the name "value" contains the values of the parameters.
        LB : [list]
            lower bound of the parameter.
        UB : [list]
            upper bound of the parameter.
        Function : TYPE
            DESCRIPTION.
        Positions : [list], optional
            position of the parameter in the list (the beginning of the list starts
            with 0), if the Position argument is empty list the sensitivity will
            be done for all parameters. The default is [].
        NoValues : [integer], optional
            number of parameter values between the bounds you want to calculate the
            metric for, if the values does not include the value if the given parameter
            it will be appended to the values. The default is 5.
        Type : [integer], optional
            type equals 1 if the function resurns one value (the measured metric)
            Type equals 2 if the function resurns two values (the measured metric,
            and any calculated values you want to check how they change by changing
            the value of the parameter). The default is 1.

        Returns
        -------
        None.

        """
        self.Parameter = Parameter
        self.LB = LB
        self.UB = UB

        assert (
            len(self.Parameter) == len(self.LB) == len(self.UB)
        ), "Length of the boundary shoulf be of the same length as the length of the parameters"
        assert callable(
            Function
        ), "function should be of type callable (function that takes arguments)"
        self.Function = Function

        self.NoValues = NoValues
        self.Type = Type
        # if the Position argument is empty list the sensitivity will be done for all parameters
        if Positions == []:
            self.NoPar = len(Parameter)
            self.Positions = list(range(len(Parameter)))
        else:
            self.NoPar = len(Positions)
            self.Positions = Positions

    def OAT(self, *args, **kwargs):
        """
        ======================================================================
           OAT(Parameter, LB, UB, Function,*args,**kwargs)
        ======================================================================
        OAT one-at-a-time sensitivity analysis.

        Parameters
        ----------
        Parameter : [dataframe]
            parameters dataframe including the parameters values in a column with
            name 'value' and the parameters name as index.
        LB : [List]
            parameters upper bounds.
        UB : [List]
            parameters lower bounds.
        Function : [function]
            the function you want to run it several times.
        *args : [positional argument]
            arguments of the function with the same exact names inside the function.
        **kwargs : [keyword argument]
            keyword arguments of the function with the same exact names inside the function.

        Returns
        -------
        sen : [Dictionary]
            for each parameter as a key, there is a list containing 4 lists,
            1-relative parameter values, 2-metric values, 3-Real parameter values
            4- adition calculated values from the function if you choose Type=2.

        """

        self.sen = {}

        for i in range(self.NoPar):
            k = self.Positions[i]
            if self.Type == 1:
                self.sen[self.Parameter.index[k]] = [[], [], []]
            else:
                self.sen[self.Parameter.index[k]] = [[], [], [], []]
            # generate 5 random values between the high and low parameter bounds
            rand_value = np.linspace(self.LB[k], self.UB[k], self.NoValues)
            # add the value of the calibrated parameter and sort the values
            rand_value = np.sort(np.append(rand_value, self.Parameter["value"][k]))
            # store the relative values of the parameters in the first list in the dict
            self.sen[self.Parameter.index[k]][0] = [
                ((h) / self.Parameter["value"][k]) for h in rand_value
            ]

            Randpar = self.Parameter["value"].tolist()
            for j in range(len(rand_value)):
                Randpar[k] = rand_value[j]
                # args = list(args)
                # args.insert(Position,Randpar)
                if self.Type == 1:
                    metric = self.Function(Randpar, *args, **kwargs)
                else:
                    metric, CalculatedValues = self.Function(Randpar, *args, **kwargs)
                    self.sen[self.Parameter.index[k]][3].append(CalculatedValues)
                try:
                    # store the metric value in the second list in the dict
                    self.sen[self.Parameter.index[k]][1].append(round(metric, 3))
                except TypeError:
                    message = """the Given Function returns more than one value,
                    the function should return only one value for Type=1, or
                    two values for Type=2.
                    """
                    assert False, message
                # store the real values of the parameter in the third list in the dict
                self.sen[self.Parameter.index[k]][2].append(round(rand_value[j], 4))
                print(str(k) + "-" + self.Parameter.index[k] + " -" + str(j))
                print(round(metric, 3))

    def Sobol(
        self,
        RealValues=False,
        Title="",  # CalculatedValues=False,
        xlabel="xlabel",
        ylabel="Metric values",
        labelfontsize=12,
        From="",
        To="",
        Title2="",
        xlabel2="xlabel2",
        ylabel2="ylabel2",
        spaces=[None, None, None, None, None, None],
    ):
        """
        =============================================================================
             Sobol(RealValues=False, CalculatedValues=False, Title='',
                   xlabel='xlabel', ylabel='Metric values', labelfontsize=12,
                   From='', To='',Title2='', xlabel2='xlabel2', ylabel2='ylabel2',
                   spaces=[None,None,None,None,None,None])
        =============================================================================

        Parameters
        ----------
        RealValues : [bool], optional
            if you want to plot the real values in the x-axis not the relative
            values, works properly only if you are checking the sensitivity of
            one parameter as the range of parameters differes. The default is False.
        CalculatedValues : [bool], optional
            if you choose Type=2 in the OAT method, then the function returns
            calculated values, and here you can True to plot it . The default is False.
        Title : [string], optional
            DESCRIPTION. The default is ''.
        xlabel : [string], optional
            DESCRIPTION. The default is 'xlabel'.
        ylabel : [string], optional
            DESCRIPTION. The default is 'Metric values'.
        labelfontsize : [integer], optional
            DESCRIPTION. The default is 12.
        From : TYPE, optional
            the calculated values are in array type and From attribute is from
            where the plotting will start. The default is ''.
        To : TYPE, optional
            the calculated values are in array type and To attribute is from
            where the plotting will end. The default is ''.
        Title2 : TYPE, optional
            DESCRIPTION. The default is ''.
        xlabel2 : TYPE, optional
            DESCRIPTION. The default is 'xlabel2'.
        ylabel2 : TYPE, optional
            DESCRIPTION. The default is 'ylabel2'.
        spaces : TYPE, optional
            DESCRIPTION. The default is [None,None,None,None,None,None].

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """

        if self.Type == 1:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

            for i in range(self.NoPar):
                k = self.Positions[i]
                if RealValues:
                    ax.plot(
                        self.sen[self.Parameter.index[k]][2],
                        self.sen[self.Parameter.index[k]][1],
                        Vis.MarkerStyle(k),
                        linewidth=3,
                        markersize=10,
                        label=self.Parameter.index[k],
                    )
                else:
                    ax.plot(
                        self.sen[self.Parameter.index[k]][0],
                        self.sen[self.Parameter.index[k]][1],
                        Vis.MarkerStyle(k),
                        linewidth=3,
                        markersize=10,
                        label=self.Parameter.index[k],
                    )

            ax.set_title(Title, fontsize=12)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            ax.tick_params(axis="both", which="major", labelsize=labelfontsize)

            ax.legend(fontsize=12)
            plt.tight_layout()
            return fig, ax
        else:  # self.Type == 2 and CalculatedValues
            try:
                fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6))

                for i in range(self.NoPar):
                    # for i in range(len(self.sen[self.Parameter.index[0]][0])):
                    k = self.Positions[i]
                    if RealValues:
                        ax1.plot(
                            self.sen[self.Parameter.index[k]][2],
                            self.sen[self.Parameter.index[k]][1],
                            Vis.MarkerStyle(k),
                            linewidth=3,
                            markersize=10,
                            label=self.Parameter.index[k],
                        )
                    else:
                        ax1.plot(
                            self.sen[self.Parameter.index[k]][0],
                            self.sen[self.Parameter.index[k]][1],
                            Vis.MarkerStyle(k),
                            linewidth=3,
                            markersize=10,
                            label=self.Parameter.index[k],
                        )

                ax1.set_title(Title, fontsize=12)
                ax1.set_xlabel(xlabel, fontsize=12)
                ax1.set_ylabel(ylabel, fontsize=12)
                ax1.tick_params(axis="both", which="major", labelsize=labelfontsize)

                ax1.legend(fontsize=12)

                for i in range(self.NoPar):
                    k = self.Positions[i]
                    # for j in range(self.NoValues):
                    for j in range(len(self.sen[self.Parameter.index[k]][0])):
                        if From == "":
                            From = 0
                        if To == "":
                            To = len(self.sen[self.Parameter.index[k]][3][j].values)

                        ax2.plot(
                            self.sen[self.Parameter.index[k]][3][j].values[From:To],
                            label=self.sen[self.Parameter.index[k]][2][j],
                        )

                # ax2.legend(fontsize=12)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(loc=6, fancybox=True, bbox_to_anchor=(1.015, 0.5))

                ax2.set_title(Title2, fontsize=12)
                ax2.set_xlabel(xlabel2, fontsize=12)
                ax2.set_ylabel(ylabel2, fontsize=12)

                plt.subplots_adjust(
                    left=spaces[0],
                    bottom=spaces[1],
                    right=spaces[2],
                    top=spaces[3],
                    wspace=spaces[4],
                    hspace=spaces[5],
                )

            except ValueError:
                assert (
                    False
                ), "to plot Calculated Values you should choose Type==2 in the sentivivity object"

            plt.tight_layout()
            return fig, (ax1, ax2)

    def ListAttributes(self):
        """
        Print Attributes List
        """

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
