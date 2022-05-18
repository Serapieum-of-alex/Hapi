"""
L moments
"""
import numpy as np
import scipy as sp

ninf = 1e-5


class Lmoments:
    """
    Hosking (1990) introduced the concept of L-moments, which are quantities that can
    be directly interpreted as scale and shape descriptors of probability distributions
    The L-moments of order r, denoted by λr

    λ1 = α0 = β0
    λ2 = α0 - 2α1 = 2β1 - β0
    λ3 = α0 - 6α1 + 6α2 = 6β2 - 6β1 + β0
    λ4 = α0 - 12α1 + 30α2 - 20α3 = 20β3 - 30β2 + 12β1 - β0

    """

    def __init__(self, data):
        self.data = data
        pass

    def Lmom(self, nmom=5):
        if nmom <= 5:
            var = self.samlmusmall(nmom)
            return var
        else:
            var = self.samlmularge(nmom)
            return var

    @staticmethod
    def comb(N, k):
        if (k > N) or (N < 0) or (k < 0):
            return 0
        val = 1
        for j in range(min(k, N - k)):
            val = (val * (N - j)) // (j + 1)
        return val

    def samlmularge(self, nmom=5):

        x = self.data
        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        x = sorted(x)
        n = len(x)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        ##Calculate first order
        ##Pretty efficient, no loops
        coefl1 = 1.0 / self.comb(n, 1)
        suml1 = sum(x)
        l = [coefl1 * suml1]

        if nmom == 1:
            return l[0]

        # Setup comb table, where comb[i][x] refers to comb(x,i)
        comb = []
        for i in range(1, nmom):
            comb.append([])
            for j in range(n):
                comb[-1].append(self.comb(j, i))

        for mom in range(2, nmom + 1):
            ##        print(mom)
            coefl = 1.0 / mom * 1.0 / self.comb(n, mom)
            xtrans = []
            for i in range(0, n):
                coeftemp = []
                for j in range(0, mom):
                    coeftemp.append(1)

                for j in range(0, mom - 1):
                    coeftemp[j] = coeftemp[j] * comb[mom - j - 2][i]

                for j in range(1, mom):
                    coeftemp[j] = coeftemp[j] * comb[j - 1][n - i - 1]

                for j in range(0, mom):
                    coeftemp[j] = coeftemp[j] * self.comb(mom - 1, j)

                for j in range(0, int(0.5 * mom)):
                    coeftemp[j * 2 + 1] = -coeftemp[j * 2 + 1]
                coeftemp = sum(coeftemp)
                xtrans.append(x[i] * coeftemp)

            if mom > 2:
                l.append(coefl * sum(xtrans) / l[1])
            else:
                l.append(coefl * sum(xtrans))
        return l

    def samlmusmall(self, nmom=5):
        x = self.data

        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        x = sorted(x)
        n = len(x)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        ##Pretty efficient, no loops
        coefl1 = 1.0 / self.comb(n, 1)

        suml1 = sum(x)
        l1 = coefl1 * suml1

        if nmom == 1:
            ret = l1
            return ret

        # comb terms appear elsewhere, this will decrease calc time
        # for nmom > 2, and shouldn't decrease time for nmom == 2
        # comb(x,1) = x
        # for i in range(1,n+1):
        ##        comb1.append(comb(i-1,1))
        ##        comb2.append(comb(n-i,1))
        # Can be simplifed to comb1 = range(0,n)

        comb1 = range(0, n)
        comb2 = range(n - 1, -1, -1)

        coefl2 = 0.5 * 1.0 / self.comb(n, 2)
        xtrans = []
        for i in range(0, n):
            coeftemp = comb1[i] - comb2[i]
            xtrans.append(coeftemp * x[i])

        l2 = coefl2 * sum(xtrans)

        if nmom == 2:
            ret = [l1, l2]
            return ret

        ##Calculate Third order
        # comb terms appear elsewhere, this will decrease calc time
        # for nmom > 2, and shouldn't decrease time for nmom == 2
        # comb3 = comb(i-1,2)
        # comb4 = comb3.reverse()
        comb3 = []
        comb4 = []
        for i in range(0, n):
            combtemp = self.comb(i, 2)
            comb3.append(combtemp)
            comb4.insert(0, combtemp)

        coefl3 = 1.0 / 3 * 1.0 / self.comb(n, 3)
        xtrans = []
        for i in range(0, n):
            coeftemp = comb3[i] - 2 * comb1[i] * comb2[i] + comb4[i]
            xtrans.append(coeftemp * x[i])

        l3 = coefl3 * sum(xtrans) / l2

        if nmom == 3:
            ret = [l1, l2, l3]
            return ret

        ##Calculate Fourth order
        # comb5 = comb(i-1,3)
        # comb6 = comb(n-i,3)
        comb5 = []
        comb6 = []
        for i in range(0, n):
            combtemp = self.comb(i, 3)
            comb5.append(combtemp)
            comb6.insert(0, combtemp)

        coefl4 = 1.0 / 4 * 1.0 / self.comb(n, 4)
        xtrans = []
        for i in range(0, n):
            coeftemp = (
                comb5[i] - 3 * comb3[i] * comb2[i] + 3 * comb1[i] * comb4[i] - comb6[i]
            )
            xtrans.append(coeftemp * x[i])

        l4 = coefl4 * sum(xtrans) / l2

        if nmom == 4:
            ret = [l1, l2, l3, l4]
            return ret

        ##Calculate Fifth order
        comb7 = []
        comb8 = []
        for i in range(0, n):
            combtemp = self.comb(i, 4)
            comb7.append(combtemp)
            comb8.insert(0, combtemp)

        coefl5 = 1.0 / 5 * 1.0 / self.comb(n, 5)
        xtrans = []
        for i in range(0, n):
            coeftemp = (
                comb7[i]
                - 4 * comb5[i] * comb2[i]
                + 6 * comb3[i] * comb4[i]
                - 4 * comb1[i] * comb6[i]
                + comb8[i]
            )
            xtrans.append(coeftemp * x[i])

        l5 = coefl5 * sum(xtrans) / l2

        if nmom == 5:
            ret = [l1, l2, l3, l4, l5]
            return ret

    @staticmethod
    def GEV(xmom):
        """
        Estimate the generalized extreme value distribution parameters using Lmoments method
        """
        eps = 1e-6
        maxit = 20
        # euler constant
        EU = 0.57721566
        DL2 = np.log(2)
        DL3 = np.log(3)
        A0 = 0.28377530
        A1 = -1.21096399
        A2 = -2.50728214
        A3 = -1.13455566
        A4 = -0.07138022
        B1 = 2.06189696
        B2 = 1.31912239
        B3 = 0.25077104
        C1 = 1.59921491
        C2 = -0.48832213
        C3 = 0.01573152
        D1 = -0.64363929
        D2 = 0.08985247

        T3 = xmom[2]
        # if std <= 0 or third moment > 1
        if xmom[1] <= 0 or abs(T3) >= 1:
            raise ValueError("L-Moments Invalid")

        if T3 <= 0:
            G = (A0 + T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * A4)))) / (
                1 + T3 * (B1 + T3 * (B2 + T3 * B3))
            )
            if T3 >= -0.8:
                shape = G
                GAM = np.exp(sp.special.gammaln(1 + G))
                scale = xmom[1] * G / (GAM * (1 - 2 ** (-G)))
                loc = xmom[0] - scale * (1 - GAM) / G
                para = [shape, loc, scale]
                return para

            if T3 <= -0.97:
                G = 1 - np.log(1 + T3) / DL2

            T0 = (T3 + 3) * 0.5
            for IT in range(1, maxit):
                X2 = 2 ** (-G)
                X3 = 3 ** (-G)
                XX2 = 1 - X2
                XX3 = 1 - X3
                T = XX3 / XX2
                DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2 ** 2)
                GOLD = G
                G = G - (T - T0) / DERIV
                if abs(G - GOLD) <= eps * G:
                    shape = G
                    GAM = np.exp(sp.special.gammaln(1 + G))
                    scale = xmom[1] * G / (GAM * (1 - 2 ** (-G)))
                    loc = xmom[0] - scale * (1 - GAM) / G
                    para = [shape, loc, scale]
                    return para

            print("Iteration has not converged")

        Z = 1 - T3
        G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))
        if abs(G) < ninf:
            scale = xmom[1] / DL2
            loc = xmom[0] - EU * scale
            para = [0, loc, scale]
            return para
        else:
            shape = G
            GAM = np.exp(sp.special.gammaln(1 + G))
            scale = xmom[1] * G / (GAM * (1 - 2 ** (-G)))
            loc = xmom[0] - scale * (1 - GAM) / G
            para = [shape, loc, scale]
            return para

    @staticmethod
    def Gumbel(mom):
        EU = 0.577215664901532861
        if mom[1] <= 0:
            raise ValueError("L-Moments Invalid")
        else:
            para2 = mom[1] / np.log(2)
            para1 = mom[0] - EU * para2
            para = [para1, para2]
            return para
