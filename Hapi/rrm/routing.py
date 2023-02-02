"""Routing File contains all Routing functions.

1- Muskingum cunge
2- triangular (MAXBAS) routing function

Created on Sun Apr 29 17:42:04 2018

@author: Mostafa
"""

from typing import Union

import numpy as np


class Routing:
    """Routing class contains routing method.

    Methods
    -------
    1- Muskingum
    2- Muskingum_V
    3- TriangularRouting1
        functions :
        1- CalculateWeights
    4- TriangularRouting2
        functions
        1- Tf
    """

    def __init__(self):
        """Routing model does not need any parameters to be instantiated."""
        pass

    @staticmethod
    def Muskingum(inflow, Qinitial, k, x, dt):
        """Muskingum.

        Parameters
        ----------
        inflow: [numpy array]
            time series of inflow hydrograph
        Qinitial: [numeric]
            initial value for outflow
        k: [numeric]
            travelling time (hours)
        x: [numeric]
            surface nonlinearity coefficient (0,0.5)
        dt: [numeric]
            delta t

        Returns
        -------
        outflow: [numpy array]
            time series of routed hydrograph

        Examples
        --------
        >>> q = [] # discharge time series
        >>> time_resolution = 1  # hourly time step
        >>> q_routed = Routing.Muskingum(q, q[0], k, x, time_resolution)
        """
        c1 = (dt - 2 * k * x) / (2 * k * (1 - x) + dt)
        c2 = (dt + 2 * k * x) / (2 * k * (1 - x) + dt)
        c3 = (2 * k * (1 - x) - dt) / (2 * k * (1 - x) + dt)

        #    if c1+c2+c3!=1:
        #        raise("sim of c1,c2 & c3 is not 1")

        outflow = np.zeros_like(inflow)
        outflow[0] = Qinitial

        for i in range(1, len(inflow)):
            outflow[i] = c1 * inflow[i] + c2 * inflow[i - 1] + c3 * outflow[i - 1]

        outflow = np.round(outflow, 4)

        return outflow

    @staticmethod
    def Muskingum_V(
        inflow: np.ndarray,
        Qinitial: Union[int, float],
        k: Union[int, float],
        x: Union[int, float],
        dt: Union[int, float],
    ) -> np.ndarray:
        """Muskingum_V.

            Vectorized version of Muskingum

        Parameters
        ----------
        inflow: [numpy array]
            time series of inflow hydrograph
        Qinitial: [numeric]
            initial value for outflow
        k: [numeric]
            travelling time (hours)
        x: [numeric]
            surface nonlinearity coefficient (0,0.5)
        dt: [numeric]
            delta t

        Returns
        -------
        outflow:
            [numpy array] time series of routed hydrograph

        Examples
        --------
        >>> q = [] # discharge time series
        >>> time_resolution = 1  # hourly time step
        >>> q_routed = Routing.Muskingum_V(q, q[0], k, x, time_resolution)
        """
        c1 = (dt - 2 * k * x) / (2 * k * (1 - x) + dt)
        c2 = (dt + 2 * k * x) / (2 * k * (1 - x) + dt)
        c3 = (2 * k * (1 - x) - dt) / (2 * k * (1 - x) + dt)

        #    if c1+c2+c3!=1:
        #        raise("sim of c1,c2 & c3 is not 1")

        Q = np.zeros_like(inflow)
        Q[0] = Qinitial
        Q[1:] = c1 * np.asarray(inflow[1:]) + c2 * np.asarray(inflow[0:-1])

        for i in range(1, len(inflow)):
            # only if the
            if not Q[i] + c3 * Q[i - 1] < 0:
                Q[i] = Q[i] + c3 * Q[i - 1]

        return Q

    @staticmethod
    def Tf(maxbas):
        """Tf.

            Transfer function weight generator in a shape of a triangle.

        Parameters
        ----------
        maxbas: [integer]
            number of time steps that the triangular routing function
            is going to divide the discharge into, based on the weights
            generated from this function, min value is 1 and default value is 1

        Returns
        -------
        wi: [numpy array]
            array of normalised weights

        Examples
        --------
        >>> ws = Routing.Tf(5)
        """

        wi = []
        for x in range(1, maxbas + 1):  # if maxbas=3 so x=[1,2,3]
            if (
                x <= (maxbas) / 2.0
            ):  # x <= 1.5  # half of values will form the rising limb and half falling limb
                # Growing transfer    # rising limb
                wi.append((x) / (maxbas + 2.0))
            else:
                # Receding transfer    # falling limb
                wi.append(1.0 - (x + 1) / (maxbas + 2.0))

        # Normalise weights
        wi = np.array(wi) / np.sum(wi)
        return wi

    @staticmethod
    def TriangularRouting2(q, maxbas=1):
        """Triangular Routing.

            The function implements the transfer function using a triangular function (considers only integer values of
            Maxbas parameter)

        Parameters
        ----------
        q: [numpy array]
            time series of discharge hydrographs
        maxbas: [integer]
            number of time steps that the triangular routing function
            is going to divide the discharge into, based on the weights
            generated from this function, min value is 1 and default value is 1

        Returns
        -------
        q_r: [numpy array]
            time series of routed hydrograph

        Examples
        --------
        >>> q_sim = Routing.TriangularRouting2(np.array(q_sim), parameters[-1])
        """
        # input data validation
        assert maxbas >= 1, "Maxbas value has to be larger than 1"

        # Get integer part of maxbas
        maxbas = int(round(maxbas, 0))

        # get the weights
        w = Routing.Tf(maxbas)

        # rout the discharge signal
        q_r = np.zeros_like(q, dtype="float64")
        q_temp = np.float32(q)
        for w_i in w:
            q_r += q_temp * w_i
            q_temp = np.insert(q_temp, 0, 0.0)[:-1]

        return q_r

    @staticmethod
    def CalculateWeights(MAXBAS):
        """Calculate Weights.

            - calculate the MAXBAS Weights based on a MAXBAX number The MAXBAS is a HBV parameter that
            controls the routing.
            - It is important to mention that this function allows to obtain weights
            not only for interger values but from decimals values as well.

        Parameters
        ----------
        MAXBAS: [Numeric]

        Examples
        --------
        >>> maxbasW = Routing.CalculateWeights(5)
        >>> print(maxbasW)
        >>> 0.0800    0.2400    0.3600    0.2400    0.0800
        """
        yant = 0
        Total = 0  # Just to verify how far from the unit is the result

        TotalA = (MAXBAS * MAXBAS * np.sin(np.pi / 3)) / 2
        IntPart = np.floor(MAXBAS)
        RealPart = MAXBAS - IntPart
        PeakPoint = MAXBAS % 2
        flag = 1  # 1 = "up"  ; 2 = down

        if RealPart > 0:  # even number 2,4,6,8,10
            maxbasW = np.ones(int(IntPart) + 1)  # if even add 1
        else:  # odd number
            maxbasW = np.ones(int(IntPart))

        for x in range(int(MAXBAS)):

            if x < (MAXBAS / 2.0) - 1:
                # Integral of  x dx with slope of 60 degree Equilateral triangle
                ynow = np.tan(np.pi / 3) * (x + 1)
                # ' Area / Total Area
                maxbasW[x] = ((ynow + yant) / 2) / TotalA
            else:  # The area here is calculated by the formlua of a trapezoidal (B1+B2)*h /2
                if flag == 1:
                    ynow = np.sin(np.pi / 3) * MAXBAS
                    if PeakPoint == 0:
                        maxbasW[x] = ((ynow + yant) / 2) / TotalA
                    else:
                        A1 = ((ynow + yant) / 2) * (MAXBAS / 2.0 - x) / TotalA
                        yant = ynow
                        ynow = (MAXBAS * np.sin(np.pi / 3)) - (
                            np.tan(np.pi / 3) * (x + 1 - MAXBAS / 2.0)
                        )
                        A2 = ((ynow + yant) * (x + 1 - MAXBAS / 2.0) / 2) / TotalA
                        maxbasW[x] = A1 + A2

                    flag = 2
                else:
                    # 'sum of the two height in the descending part of the triangle
                    ynow = MAXBAS * np.sin(np.pi / 3) - np.tan(np.pi / 3) * (
                        x + 1 - MAXBAS / 2.0
                    )
                    # Multiplying by the height of the trapezoidal and dividing by 2
                    maxbasW[x] = ((ynow + yant) / 2) / TotalA

            Total = Total + maxbasW[x]
            yant = ynow

        x = int(MAXBAS)
        # x = x + 1

        if RealPart > 0:
            if np.floor(MAXBAS) == 0:
                MAXBAS = 1
                maxbasW[x] = 1
                NumberofWeights = 1
            else:
                maxbasW[x] = (yant * (MAXBAS - (x)) / 2) / TotalA
                Total = Total + maxbasW[x]
                NumberofWeights = x
        else:
            NumberofWeights = x - 1

        return maxbasW

    @staticmethod
    def TriangularRouting1(Q, MAXBAS):
        """TriangularRouting1.

        calculate the routing from a input hydrograph using the MAXBAS parameter from the HBV
        model (considers float values of Maxbas parameter).

        Examples
        --------
            [Qout,maxbasW]=RoutingMAXBAS(Q,5);
            where:
            Qout = output hydrograph
            maxbasW = MAXBAS weight
            Q = input hydrograph
            5 = MAXBAS parameter value.
        """
        # CALCULATE MAXBAS WEIGHTS
        maxbasW = Routing.CalculateWeights(MAXBAS)

        Qw = np.ones((len(Q), len(maxbasW)))
        # Calculate the matrix discharge
        for i in range(len(Q)):  # 0 to 10
            for k in range(len(maxbasW)):  # 0 to 4
                Qw[i, k] = maxbasW[k] * Q[i]

        def mm(A, s):
            tot = []
            for o in range(np.shape(A)[1]):  # columns
                for t in range(np.shape(A)[0]):  # rows
                    tot.append(A[t, o])
            Su = tot[s:-1:s]
            return Su

        # Calculate routing
        j = 0
        Qout = np.ones(shape=(len(Q)))

        for i in range(len(Q)):
            if i == 0:
                Qout[i] = Qw[i, i]
            elif i < len(maxbasW) - 1:
                A = Qw[0 : i + 1, :]
                s = len(A) - 1  # len(A) is the no of rows or use int(np.shape(A)[0])
                Su = mm(A, s)

                Qout[i] = sum(Su[0 : i + 1])
            else:
                A = Qw[j : i + 1, :]
                s = len(A) - 1
                Su = mm(A, s)
                Qout[i] = sum(Su)
                j = j + 1

        return Qout  # ,maxbasW
