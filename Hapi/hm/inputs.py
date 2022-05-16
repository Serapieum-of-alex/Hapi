"""
Created on Sat Feb 15 22:51:02 2020

@author: mofarrag
"""
import datetime as dt
import os
import zipfile
from typing import Union
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.stats import gumbel_r  # genextreme,

from Hapi.hm.river import River
from Hapi.sm.distributions import GEV, Gumbel, PlottingPosition

# from matplotlib import gridspec


class Inputs(River):
    """Inputs.

    Methods
        1- ExtractHydrologicalInputs
        2- StatisticalProperties
        3- WriteHQFile
        4- ReturnPeriod
        5- ReadRIMResult
        6- CreateTraceALL
    """

    def __init__(self, Name, version=3):
        self.Name = Name
        self.version = version
        self.StatisticalPr = None
        self.DistributionPr = None

    def ExtractHydrologicalInputs(
        self, WeatherGenerator, FilePrefix, realization, path, SWIMNodes, SavePath
    ):
        """ExtractHydrologicalInputs.

        Parameters
        ----------
            WeatherGenerator : TYPE
                DESCRIPTION.
            FilePrefix : TYPE
                DESCRIPTION.
            realization : [Integer]
                type the number of the realization (the order of the 100 year
                run by swim).
            path : [String]
                 SWIMResultFile is the naming format you used in naming the result
                 files of the discharge values stored with the name of the file as
                 out+realization number + .dat (ex out15.dat).
            SWIMNodes : [String]
                text file containing the list of sub-basins IDs or computational nodes ID you
                have used to run SWIM and store the results.
            SavePath : [String]
                path to the folder where you want to save the separate file for
                each sub-basin.

        Returns
        -------
        None.

        """
        if WeatherGenerator:
            """ WeatherGenerator """
            SWIMResultFile = FilePrefix + str(realization) + ".dat"
            # 4-5
            # check whether the the name of the realization the same as the name of 3
            # the saving file or not to prevent any confusion in saving the files
            if int(realization) <= 9:
                assert int(SWIMResultFile[-5:-4]) == int(
                    SavePath[-1]
                ), " Wrong files sync "
            else:
                assert int(SWIMResultFile[-6:-4]) == int(
                    SavePath[-2:]
                ), " Wrong files sync "
        else:
            """ Observed data """
            SWIMResultFile = FilePrefix + str(realization) + ".dat"

        """
        SWIM writes the year as the first colmun then day as a second column and the
        discharge values starts from the thirst column so you have to write number of
        columns to be ignored at the begining
        """
        ignoreColumns = 2

        # read SWIM result file
        SWIMData = pd.read_csv(
            path + "/" + SWIMResultFile, delimiter=r"\s+", header=None
        )
        Nodes = pd.read_csv(path + "/" + SWIMNodes, header=None)

        for i in range(len(Nodes)):
            SWIMData.loc[:, i + ignoreColumns].to_csv(
                SavePath + "/" + str(Nodes.loc[i, 0]) + ".txt", header=None, index=None
            )

    def StatisticalProperties(
        self,
        ComputationalNodes: Union[str, list],
        TSdirectory: str,
        start: str,
        WarmUpPeriod: int,
        SavePlots: bool,
        SavePath: str,
        SeparateFiles: bool=False,
        Filter: Union[bool, float, int]=False,
        Distibution: str="GEV",
        method: str="lmoments",
        EstimateParameters: bool=False,
        Quartile: float=0,
        SignificanceLevel: float=0.1,
        file_extension: str = '.txt',
        date_format: str = "%Y-%m-%d",
    ):
        """StatisticalProperties.

        StatisticalProperties method reads the discharge hydrographs of rainfall
        runoff model and hydraulic model for some computational nodes and
        calculate some statistical properties.

        the code assumes that the time series are of a daily temporal resolution,
        and that the hydrological year is 1-Nov/31-Oct
        (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
        ComputationalNodes : [list]
            The file/list which contains the ID of the computational
            nodes you want to do the statistical analysis for, the ObservedFile
            should contain the discharge time series of these nodes in order.
        TSdirectory : [str]
            The directory where the time series files exist.
        start : [string]
            the begining date of the time series.
        WarmUpPeriod : [integer]
            the number of days you want to neglect at the begining of the
            Simulation (warm up period).
        SavePlots : [Bool]
            True if you want to save the plots.
        SavePath : [String]
            the path where you want to  save the statistical properties.
        SeparateFiles: [Bool]
            if the discharge data are stored in separate files not all in one file
            SeparateFiles should be True, default [False].
        Filter: [Bool]
            for observed or RIMresult data it has gaps of times where the
            model did not run or gaps in the observed data if these gap days
            are filled with a specific value and you want to ignore it here
            give Filter = Value you want
        Distibution: [str]
            Default is "GEV".
        method: [str]
            Default is "lmoments"
        EstimateParameters: [bool]
            Default is False.
        Quartile: [float]
            Default is 0.
        SignificanceLevel:
            Default is [0.1].
        file_extension: [str]
            Default is '.txt'.
        date_format: [str]
            Default is "%Y-%m-%d".

        Returns
        -------
        Statistical Properties.csv:
            file containing some statistical properties like mean, std, min, 5%, 25%,
            median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
            q100, q200, q500.
        """
        if isinstance(ComputationalNodes, str):
            ComputationalNodes = np.loadtxt(ComputationalNodes, dtype=np.uint16)
        elif isinstance(ComputationalNodes, list):
            ComputationalNodes = ComputationalNodes
        else:
            raise TypeError("ComputationalNodes should be either a path or a list")

        # hydrographs
        if SeparateFiles:
            TS = pd.DataFrame()
            # for the hydraulic model results
            logger.info("The function ignores the date column in the time series files and starts from the "
                        "given start parameter to the function so check if it is the same start date as in "
                        "the files")

            for i in range(len(ComputationalNodes)):
                TS.loc[:, int(ComputationalNodes[i])] = pd.read_csv(
                    TSdirectory + "/" + str(int(ComputationalNodes[i])) + file_extension,
                    skiprows=1, header=None,
                )[1].tolist()

            StartDate = dt.datetime.strptime(start, date_format)
            EndDate = StartDate + dt.timedelta(days=TS.shape[0] - 1)
            ind = pd.date_range(StartDate, EndDate)
            TS.index = ind
        else:
            TS = pd.read_csv(TSdirectory, delimiter=r"\s+", header=None)
            StartDate = dt.datetime.strptime(start, date_format)
            EndDate = StartDate + dt.timedelta(days=TS.shape[0] - 1)
            TS.index = pd.date_range(StartDate, EndDate, freq="D")
            # delete the first two columns
            del TS[0], TS[1]
            TS.columns = ComputationalNodes

        # neglect the first year (warmup year) in the time series
        TS = TS.loc[StartDate + dt.timedelta(days=WarmUpPeriod) : EndDate, :]

        # List of the table output, including some general data and the return periods.
        col_csv = [
            "mean",
            "std",
            "min",
            "5%",
            "25%",
            "median",
            "75%",
            "95%",
            "max",
            "t_beg",
            "t_end",
            "nyr",
        ]
        rp_name = [
            "q1.5",
            "q2",
            "q5",
            "q10",
            "q25",
            "q50",
            "q100",
            "q200",
            "q500",
            "q1000",
        ]
        col_csv = col_csv + rp_name

        # In a table where duplicates are removed (np.unique), find the number of
        # gauges contained in the .csv file.
        # no_gauge = len(ComputationalNodes)
        # Declare a dataframe for the output file, with as index the gaugne numbers
        # and as columns all the output names.
        StatisticalPr = pd.DataFrame(np.nan, index=ComputationalNodes, columns=col_csv)
        StatisticalPr.index.name = "id"
        if Distibution == "GEV":
            DistributionPr = pd.DataFrame(
                np.nan, index=ComputationalNodes, columns=["c", "loc", "scale", "D-static", "P-Value"]
            )
        else:
            DistributionPr = pd.DataFrame(
                np.nan, index=ComputationalNodes, columns=["loc", "scale", "D-static", "P-Value"]
            )
        DistributionPr.index.name = "id"
        # required return periods
        T = [1.5, 2, 5, 10, 25, 50, 50, 100, 200, 500, 1000]
        T = np.array(T)
        # these values are the Non Exceedance probability (F) of the chosen
        # return periods F = 1 - (1/T)
        # Non Exceedance propabilities
        # F = [1/3, 0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.995, 0.998]
        F = 1 - (1 / T)
        # Iteration over all the gauge numbers.
        if SavePlots:
            rpath = os.path.join(SavePath, "Figures")
            if not os.path.exists(rpath):
                os.mkdir(rpath)

        for i in ComputationalNodes:
            QTS = TS.loc[:, i]
            # The time series is resampled to the annual maxima, and turned into a
            # numpy array.
            # The hydrological year is 1-Nov/31-Oct (from Petrow and Merz, 2009, JoH).
            amax = QTS.resample("A-OCT").max().values

            if type(Filter) != bool:
                amax = amax[amax != Filter]

            if EstimateParameters:
                # TODO: still to be tested and prepared for GEV
                # estimate the parameters through an optimization
                # alpha = (np.sqrt(6) / np.pi) * amax.std()
                # beta = amax.mean() - 0.5772 * alpha
                # param_dist = [beta, alpha]
                threshold = np.quantile(amax, Quartile)
                if Distibution == "GEV":
                    dist = GEV(amax)
                    param_dist = dist.EstimateParameter(
                        method='optimization',
                        ObjFunc=Gumbel.ObjectiveFn,
                        threshold=threshold,
                    )
                else:
                    dist = Gumbel(amax)
                    param_dist = dist.EstimateParameter(
                        method='optimization',
                        ObjFunc=Gumbel.ObjectiveFn,
                        threshold=threshold,
                    )
            else:
                # estimate the parameters through an maximum liklehood method
                if Distibution == "GEV":
                    dist = GEV(amax)
                    # defult parameter estimation method is maximum liklihood method
                    param_dist = dist.EstimateParameter(method=method)
                else:
                    # A gumbel distribution is fitted to the annual maxima
                    dist = Gumbel(amax)
                    # defult parameter estimation method is maximum liklihood method
                    param_dist = dist.EstimateParameter(method=method)

            DistributionPr.loc[i,"D-static"], DistributionPr.loc[i,"P-Value"] = dist.ks()
            if Distibution == "GEV":
                DistributionPr.loc[i, "c"] = param_dist[0]
                DistributionPr.loc[i, "loc"] = param_dist[1]
                DistributionPr.loc[i, "scale"] = param_dist[2]
            else:
                DistributionPr.loc[i, "loc"] = param_dist[0]
                DistributionPr.loc[i, "scale"] = param_dist[1]

            # Return periods from the fitted distribution are stored.
            # get the Discharge coresponding to the return periods
            if Distibution == "GEV":
                Qrp = dist.TheporeticalEstimate(
                    param_dist[0], param_dist[1], param_dist[2], F
                )
            else:
                Qrp = dist.TheporeticalEstimate(param_dist[0], param_dist[1], F)

            # to get the Non Exceedance probability for a specific Value
            # sort the amax
            amax.sort()
            # calculate the F (Exceedence probability based on weibul)
            cdf_Weibul = PlottingPosition.Weibul(amax)
            # Gumbel.ProbapilityPlot method calculates the theoretical values
            # based on the Gumbel distribution
            # parameters, theoretical cdf (or weibul), and calculate the confidence interval
            if SavePlots:

                if Distibution == "GEV":
                    fig, ax = dist.ProbapilityPlot(param_dist[0], param_dist[1], param_dist[2], cdf_Weibul,
                                                   alpha=SignificanceLevel)
                else:
                    fig, ax = dist.ProbapilityPlot(
                        param_dist[0], param_dist[1], cdf_Weibul, alpha=SignificanceLevel
                    )

                fig[0].savefig(f"{SavePath}/Figures/{i}.png", format="png")
                plt.close()

                fig[1].savefig(f"{SavePath}/Figures/F-{i}.png", format="png")
                plt.close()

            StatisticalPr.loc[i, "mean"] = QTS.mean()
            StatisticalPr.loc[i, "std"] = QTS.std()
            StatisticalPr.loc[i, "min"] = QTS.min()
            StatisticalPr.loc[i, "5%"] = QTS.quantile(0.05)
            StatisticalPr.loc[i, "25%"] = QTS.quantile(0.25)
            StatisticalPr.loc[i, "median"] = QTS.quantile(0.50)
            StatisticalPr.loc[i, "75%"] = QTS.quantile(0.75)
            StatisticalPr.loc[i, "95%"] = QTS.quantile(0.95)
            StatisticalPr.loc[i, "max"] = QTS.max()
            StatisticalPr.loc[i, "t_beg"] = QTS.index.min()
            StatisticalPr.loc[i, "t_end"] = QTS.index.max()
            StatisticalPr.loc[i, "nyr"] = (
                StatisticalPr.loc[i, "t_end"] - StatisticalPr.loc[i, "t_beg"]
            ).days / 365.25
            for irp, irp_name in zip(Qrp, rp_name):
                StatisticalPr.loc[i, irp_name] = irp

            # Print for prompt and check progress.
            print("Gauge", i, "done.")

        # Output file
        StatisticalPr.to_csv(f"{SavePath}/Statistical Properties.csv", float_format="%.4f")
        self.StatisticalPr = StatisticalPr
        DistributionPr.to_csv(f"{SavePath}/DistributionProperties.csv" , float_format="%.4f")
        self.DistributionPr = DistributionPr


    def WriteHQFile(self, NoNodes: int, StatisticalPropertiesFile:str, SaveTo:str):
        """WriteHQFile.

        :param NoNodes:
        :param StatisticalPropertiesFile:
        :param SaveTo:
        :return:
        """
        # Create a table of all nodes and sub-basins
        HQ = pd.DataFrame(
            index=list(range(NoNodes)), columns=["id", "2yrs", "10yrs", "100yrs"]
        )
        HQ.loc[:, ["2yrs", "10yrs", "100yrs"]] = -1
        HQ.loc[:, "id"] = range(1, NoNodes + 1)
        StatisticalPr = pd.read_csv(StatisticalPropertiesFile)
        for i in range(len(StatisticalPr)):
            # i=0
            HQ.loc[StatisticalPr.loc[i, "id"] - 1, "2yrs"] = StatisticalPr.loc[i, "q2"]
            HQ.loc[StatisticalPr.loc[i, "id"] - 1, "10yrs"] = StatisticalPr.loc[
                i, "q10"
            ]
            HQ.loc[StatisticalPr.loc[i, "id"] - 1, "100yrs"] = StatisticalPr.loc[
                i, "q100"
            ]

        # save the HQ file
        HQ.to_csv(SaveTo, float_format="%%.2f".format, index=False, header=None)

    @staticmethod
    def StringSpace(Inp):
        return str(Inp) + "  "

    def ReturnPeriod(
        self,
        MapsPath,
        prefix,
        DistributionPrF,
        TraceF,
        SubsF,
        replacementF,
        HydrologicalInputsPath,
        SubIDMapF,
        ExtraSubsF,
        Fromfile,
        Tofile,
        SaveTo,
        wpath,
    ):

        AllResults = os.listdir(MapsPath)
        # list of the Max Depth files only
        MaxDepthList = list()
        for i in range(len(AllResults)):
            if AllResults[i].startswith(prefix):
                MaxDepthList.append(AllResults[i])
        # Read Inputs
        # read the Distribution parameters for each upstream computatiopnal node
        DistributionPr = pd.read_csv(DistributionPrF)
        USnode = pd.read_csv(TraceF, header=None)
        USnode.columns = ["SubID", "US", "DS"]
        # get the sub basin Id from the guide file it is the same shape in RIM1.0 and RIM2.0
        SubsID = pd.read_csv(SubsF, header=None, usecols=[0])

        ReplacementSub = pd.read_csv(replacementF)

        # read the hydrograph for all the US nodes
        # StartDate = "1950-1-1"
        # StartDate = dt.datetime.strptime(StartDate,"%Y-%m-%d")
        # ind = pd.date_range(StartDate, StartDate + dt.timedelta(days = NoYears*365), freq = "D")

        ind = range(
            1,
            len(
                pd.read_csv(
                    HydrologicalInputsPath
                    + "/"
                    + str(int(USnode.loc[SubsID.loc[10, 0] - 1, "US"]))
                    + ".txt"
                ).values
            ),
        )

        Hydrographs = pd.DataFrame(index=ind, columns=SubsID[0].to_list())

        for i in range(len(SubsID)):
            #    i=1
            # search for the SubId in the USnode or it is listed by order so subID=343 exist
            # in the row 342 (SubID-1)
            # np.where(USnode['SubID'] == SubsID.loc[i,0])
            try:
                if int(USnode.loc[SubsID.loc[i, 0] - 1, "US"]) != -1:
                    Hydrographs.loc[:, SubsID.loc[i, 0]] = pd.read_csv(
                        HydrologicalInputsPath
                        + "/"
                        + str(int(USnode.loc[SubsID.loc[i, 0] - 1, "US"]))
                        + ".txt"
                    ).values[: len(Hydrographs)]
            except:
                OtherSubLoc = np.where(ReplacementSub["missing"] == SubsID.loc[i, 0])[
                    0
                ][0]
                if (
                    int(
                        USnode.loc[
                            ReplacementSub.loc[OtherSubLoc, "replacement"] - 1, "US"
                        ]
                    )
                    != -1
                ):
                    Hydrographs.loc[:, SubsID.loc[i, 0]] = pd.read_csv(
                        HydrologicalInputsPath
                        + "/"
                        + str(
                            int(
                                USnode.loc[
                                    ReplacementSub.loc[OtherSubLoc, "replacement"] - 1,
                                    "US",
                                ]
                            )
                        )
                        + ".txt"
                    ).values[: len(Hydrographs)]

        # read sub basin map id
        SubIDMap = gdal.Open(SubIDMapF)
        SubIDMapV = SubIDMap.ReadAsArray()
        # NoValue = SubIDMap.GetRasterBand(1).GetNoDataValue()
        # SubIDMapV[SubIDMapV == NoValue] = 0
        # plt.imshow(SubIDMapV)

        # read the added subs reference text file
        ExtraSubs = pd.read_csv(ExtraSubsF)

        # function to write the numbers in the ASCII file

        # read Max depth map
        check = list()
        Klist = list()

        if Tofile == "end" or Tofile > len(MaxDepthList):
            Tofile = len(MaxDepthList)

        # Fromfile = 48
        # Tofile = Fromfile +1

        for k in range(Fromfile, Tofile):

            try:
                # open the zip file
                Compressedfile = zipfile.ZipFile(MapsPath + "/" + MaxDepthList[k])
            except:
                print("Error Opening the compressed file")
                check.append(MaxDepthList[k][len(prefix) : -4])
                Klist.append(k)
                continue

            # get the file name
            fname = Compressedfile.infolist()[0]
            # get the time step from the file name
            timestep = int(fname.filename[len(prefix) : -4])
            print("File= " + str(timestep))

            ASCIIF = Compressedfile.open(fname)
            f = ASCIIF.readlines()
            SpatialRef = f[:6]
            ASCIIRaw = f[6:]
            # ASCIIF = Compressedfile.open(fname)
            # ASCIIRaw = ASCIIF.readlines()[6:]
            rows = len(ASCIIRaw)
            cols = len(ASCIIRaw[0].split())
            MaxDepth = np.ones((rows, cols), dtype=np.float32)
            # read the ascii file
            for i in range(rows):
                x = ASCIIRaw[i].split()
                MaxDepth[i, :] = list(map(float, x))

            # check on the values of the water depth
            #    if np.shape(MaxDepth[np.isnan(MaxDepth)])[0] > 0:
            #        check.append(timestep)
            #        print("Error Check Max Depth values")
            #        continue

            # plotting to check values
            #    fromrow = np.where(MaxDepth == MaxDepth.max())[0][0]
            #    fromcol = np.where(MaxDepth == MaxDepth.max())[1][0]
            #    plt.imshow(MaxDepth[fromrow-20:fromrow+20,fromcol-20:fromcol+20])
            #    plt.imshow(MaxDepth)
            #    plt.colorbar()

            # get the Peak of the hydrograph for the whole event
            # (14 days before the end of the event)
            MaxValuedf = Hydrographs.loc[timestep - 14 : timestep, :]
            MaxValues = MaxValuedf.max().values.tolist()
            T = list()

            # Calculate the the Return period for the max Q at this time step for each
            for i in range(len(MaxValues)):
                # if the sub basin is a lateral and not routed in RIM it will not have a
                # hydrograph
                if np.isnan(MaxValues[i]):
                    T.append(np.nan)
                if not np.isnan(MaxValues[i]):
                    # np.where(USnode['SubID'] == SubsID.loc[i,0])
                    try:
                        DSnode = USnode.loc[SubsID.loc[i, 0] - 1, "US"]
                        loc = np.where(DistributionPr["id"] == DSnode)[0][0]
                    except IndexError:
                        OtherSubLoc = np.where(
                            ReplacementSub["missing"] == SubsID.loc[i, 0]
                        )[0][0]
                        DSnode = USnode.loc[
                            ReplacementSub.loc[OtherSubLoc, "replacement"] - 1, "US"
                        ]
                        loc = np.where(DistributionPr["id"] == DSnode)[0][0]

                    # to get the Non Exceedance probability for a specific Value
                    F = gumbel_r.cdf(
                        MaxValues[i],
                        loc=DistributionPr.loc[loc, "loc"],
                        scale=DistributionPr.loc[loc, "scale"],
                    )
                    # then calculate the the T (return period) T = 1/(1-F)
                    T.append(round(1 / (1 - F), 2))

            try:
                RetunPeriodMap = np.ones((rows, cols), dtype=np.float32) * 0
                for i in range(rows):
                    for j in range(cols):
                        # print("i = " + str(i) + ", j= " + str(j))
                        if not np.isnan(MaxDepth[i, j]):
                            if MaxDepth[i, j] > 0:
                                # print("i = " + str(i) + ", j= " + str(j))
                                # if the sub basin is in the Sub ID list
                                if SubIDMapV[i, j] in SubsID[0].tolist():
                                    # print("Sub = " + str(SubIDMapV[i,j]))
                                    # go get the return period directly
                                    RetunPeriodMap[i, j] = T[
                                        np.where(SubsID[0] == SubIDMapV[i, j])[0][0]
                                    ]
                                else:
                                    # print("Extra  Sub = " + str(SubIDMapV[i,j]))
                                    # the sub ID is one of the added subs not routed by RIM
                                    # so it existed in the ExtraSubs list with a reference to
                                    # a SubID routed by RIM
                                    RIMSub = ExtraSubs.loc[
                                        np.where(
                                            ExtraSubs["addSub"] == SubIDMapV[i, j]
                                        )[0][0],
                                        "RIMSub",
                                    ]
                                    RetunPeriodMap[i, j] = T[
                                        np.where(SubsID[0] == RIMSub)[0][0]
                                    ]
            except:
                print("Error")
                check.append(timestep)
                Klist.append(k)
                continue

            # save the return period ASCII file
            fname = "ReturnPeriod" + str(timestep) + ".asc"

            with open(SaveTo + "/" + fname, "w") as File:
                # write the first lines
                for i in range(len(SpatialRef)):
                    File.write(str(SpatialRef[i].decode()[:-2]) + "\n")

                for i in range(np.shape(RetunPeriodMap)[0]):
                    File.writelines(list(map(self.StringSpace, RetunPeriodMap[i, :])))
                    File.write("\n")

            # zip the file
            with zipfile.ZipFile(
                SaveTo + "/" + fname[:-4] + ".zip", "w", zipfile.ZIP_DEFLATED
            ) as newzip:
                newzip.write(SaveTo + "/" + fname, arcname=fname)
            # delete the file
            os.remove(SaveTo + "/" + fname)

        check = list(zip(check, Klist))
        if len(check) > 0:
            np.savetxt(wpath + "CheckWaterDepth.txt", check, fmt="%6d")


    def CreateTraceALL(
        self,
        ConfigFilePath,
        RIMSubsFilePath,
        TraceFile,
        USonly=1,
        HydrologicalInputsFile="",
    ):
        """CreateTraceALL.

        Parameters
        ----------
        ConfigFilePath : [String]
            SWIM configuration file.
        RIMSubsFilePath : [String]
            path to text file with all the ID of SWIM sub-basins.
        TraceFile : TYPE
            DESCRIPTION.
        USonly : TYPE, optional
            DESCRIPTION. The default is 1.
        HydrologicalInputsFile : TYPE, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.

        """
        # reading the file
        Config = pd.read_csv(ConfigFilePath, header=None)
        # process the Configuration file
        # get the Route rows from the file
        Route = pd.DataFrame(columns=["No", "DSnode", "SubID", "USnode", "No2"])

        j = 0
        for i in range(len(Config)):
            if Config[0][i].split()[0] == "route":
                Route.loc[j, :] = list(map(int, Config[0][i].split()[1:]))
                j = j + 1

        # get RIM Sub-basins
        Subs = pd.read_csv(RIMSubsFilePath, header=None)
        Subs = Subs.rename(columns={0: "SubID"})


        Subs["US"] = None
        Subs["DS"] = None

        for i in range(len(Subs)):
            try:
                # if the sub-basin is in the route array so it is routed by SWIM
                loc = np.where(Route["SubID"] == Subs.loc[i, "SubID"])[0][0]
                Subs.loc[i, "US"] = int(Route.loc[loc, "USnode"])
                Subs.loc[i, "DS"] = int(Route.loc[loc, "DSnode"])
            except IndexError:
                # if the sub-basin is not in the route array so it is not routed by SWIM
                # but still can be routed using RIM

                # Subs.loc[i,'US'] = None
                # Subs.loc[i,'DS'] = None
                Subs.loc[i, "US"] = -1
                Subs.loc[i, "DS"] = -1

        # Save the file with the same format required for the hg R code

        Subs.to_csv(TraceFile, index=None, header=True)
        #    ToSave = Subs.loc[:,['SubID','US']]
        #    ToSave['Extra column 1'] = -1
        #    ToSave['Extra column 2'] = -1
        #    ToSave.to_csv(SavePath + TraceFile,header = None, index = None)

    def ListAttributes(self):
        """ListAttributes.

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
