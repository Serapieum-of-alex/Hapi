"""Created on Sat Feb 15 22:51:02 2020.

@author: mofarrag
"""
import datetime as dt
import os
import zipfile
from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
from osgeo import gdal
from scipy.stats import gumbel_r
from statista.eva import ams_analysis

from Hapi.hm.river import River

# from matplotlib import gridspec


class Inputs(River):
    """Hydraulic model Inputs class.

    Methods
    -------
        1- ExtractHydrologicalInputs
        2- StatisticalProperties
        3- WriteHQFile
        4- ReturnPeriod
        5- ReadRIMResult
        6- CreateTraceALL
    """

    def __init__(self, Name, version=3):
        """Input.

            Inputs is an object to create the inputs for the river model

        Parameters
        ----------
        Name: [str]

        version: [int]
            model version
        """
        self.Name = Name
        self.version = version
        self.statistical_properties = None
        self.distribution_properties = None

    def extractHydrologicalInputs(
        self, weather_generator, file_prefix, realization, path, locations, save_path
    ):
        """ExtractHydrologicalInputs.

        Parameters
        ----------
        weather_generator : TYPE
            DESCRIPTION.
        file_prefix : TYPE
            DESCRIPTION.
        realization : [Integer]
            type the number of the realization (the order of the 100 year
            run by swim).
        path : [String]
             rrm_result_file is the naming format you used in naming the result
             files of the discharge values stored with the name of the file as
             out+realization number + .dat (ex out15.dat).
        locations : [String]
            text file containing the list of sub-basins IDs or computational nodes ID you
            have used to run the rrm and store the results.
        save_path : [String]
            path to the folder where you want to save the separate file for
            each sub-basin.

        Returns
        -------
        None.
        """
        if weather_generator:
            """weather_generator."""
            rrm_result_file = file_prefix + str(realization) + ".dat"
            # 4-5
            # check whether the the name of the realization the same as the name of 3
            # the saving file or not to prevent any confusion in saving the files
            if int(realization) <= 9:
                assert int(rrm_result_file[-5:-4]) == int(
                    save_path[-1]
                ), " Wrong files sync "
            else:
                assert int(rrm_result_file[-6:-4]) == int(
                    save_path[-2:]
                ), " Wrong files sync "
        else:
            """Observed data."""
            rrm_result_file = file_prefix + str(realization) + ".dat"

        """
        SWIM writes the year as the first colmun then day as a second column and the
        discharge values starts from the thirst column so you have to write number of
        columns to be ignored at the begining
        """
        ignoreColumns = 2

        # read SWIM result file
        SWIMData = pd.read_csv(
            f"{path}/{rrm_result_file}", delimiter=r"\s+", header=None
        )
        Nodes = pd.read_csv(f"{path}/{locations}", header=None)

        for i in range(len(Nodes)):
            SWIMData.loc[:, i + ignoreColumns].to_csv(
                f"{save_path}/{Nodes.loc[i, 0]}.txt", header=None, index=None
            )

    def get_statistical_properties(
        self,
        gauges: list,
        rdir: str,
        start: str,
        warm_up_period: int,
        save_plots: bool,
        save_to: str,
        filter_out: Union[bool, float, int] = False,
        distribution: str = "GEV",
        method: str = "lmoments",
        quartile: float = 0,
        significance_level: float = 0.1,
        file_extension: str = ".txt",
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
        gauges : [list]
            The list which contains the ID of the gauges you want to do the statistical analysis for, the ObservedFile
            should contain the discharge time series of these nodes in order.
        rdir : [str]
            The directory where the time series files exist.
        start : [string]
            the begining date of the time series.
        warm_up_period : [int]
            The number of days you want to neglect at the begining of the
            Simulation (warm up period).
        save_plots : [Bool]
            True if you want to save the plots.
        save_to : [str]
            the rdir where you want to  save the statistical properties.
        filter_out: [Bool]
            For observed or hydraulic model data it has gaps of times where the
            model did not run or gaps in the observed data if these gap days
            are filled with a specific value and you want to ignore it here
            give filter_out = Value you want
        distribution: [str]
            Default is "GEV".
        method: [str]
            available methods are 'mle', 'mm', 'lmoments', optimization. Default is "lmoments"
        quartile: [float]
            Default is 0.
        significance_level:
            Default is [0.1].
        file_extension: [str]
            Default is '.txt'.
        date_format: [str]
            Default is "%Y-%m-%d".

        Returns
        -------
        statistical-properties.csv:
            file containing some statistical properties like mean, std, min, 5%, 25%,
            median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
            q100, q200, q500.
        """
        if not isinstance(gauges, list):
            raise TypeError("gauges should be either a rdir or a list")

        # hydrographs
        time_series = pd.DataFrame()
        # for the hydraulic model results
        logger.info(
            "The function ignores the date column in the time series files and starts from the "
            "given start parameter to the function so check if it is the same start date as in "
            "the files"
        )

        skip = []
        for gauge_i in gauges:
            try:
                time_series.loc[:, int(gauge_i)] = pd.read_csv(
                    f"{rdir}/{int(gauge_i)}{file_extension}",
                    skiprows=1,
                    header=None,
                )[1].tolist()
            except FileNotFoundError:
                logger.warning(f"File {int(gauge_i)}{file_extension} does not exist")
                skip.append(int(gauge_i))

        start_date = dt.datetime.strptime(start, date_format)
        end_date = start_date + dt.timedelta(days=time_series.shape[0] - 1)
        ind = pd.date_range(start_date, end_date)
        time_series.index = ind

        # neglect the first year (warmup year) in the time series
        time_series = time_series.loc[
            start_date + dt.timedelta(days=warm_up_period) : end_date, :
        ]

        statistical_properties, distribution_properties = ams_analysis(
            time_series,
            ams_start="A-OCT",
            save_plots=save_plots,
            save_to=save_to,
            filter_out=filter_out,
            distribution=distribution,
            method=method,
            # estimate_parameters=estimate_parameters,
            quartile=quartile,
            significance_level=significance_level,
        )

        # Output file
        statistical_properties.to_csv(
            f"{save_to}/statistical-properties.csv", float_format="%.4f"
        )

        distribution_properties.to_csv(
            f"{save_to}/DistributionProperties.csv", float_format="%.4f"
        )

        self.statistical_properties = statistical_properties
        self.distribution_properties = distribution_properties

    @staticmethod
    def StringSpace(Inp):
        """StringSpace."""
        return str(Inp) + "  "

    def return_period(
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
        save_to,
        wpath,
    ):
        """Return period."""
        AllResults = os.listdir(MapsPath)
        # list of the Max Depth files only
        MaxDepthList = list()
        for i in range(len(AllResults)):
            if AllResults[i].startswith(prefix):
                MaxDepthList.append(AllResults[i])
        # Read Inputs
        # read the Distribution parameters for each upstream computatiopnal node
        distribution_properties = pd.read_csv(DistributionPrF)
        USnode = pd.read_csv(TraceF, header=None)
        USnode.columns = ["SubID", "US", "DS"]
        # get the sub basin Id from the guide file it is the same shape in RIM1.0 and RIM2.0
        SubsID = pd.read_csv(SubsF, header=None, usecols=[0])

        ReplacementSub = pd.read_csv(replacementF)

        # read the hydrograph for all the US nodes
        # start_date = "1950-1-1"
        # start_date = dt.datetime.strptime(start_date,"%Y-%m-%d")
        # ind = pd.date_range(start_date, StartDate + dt.timedelta(days = NoYears*365), freq = "D")

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
                        loc = np.where(distribution_properties["id"] == DSnode)[0][0]
                    except IndexError:
                        OtherSubLoc = np.where(
                            ReplacementSub["missing"] == SubsID.loc[i, 0]
                        )[0][0]
                        DSnode = USnode.loc[
                            ReplacementSub.loc[OtherSubLoc, "replacement"] - 1, "US"
                        ]
                        loc = np.where(distribution_properties["id"] == DSnode)[0][0]

                    # to get the Non Exceedance probability for a specific Value
                    F = gumbel_r.cdf(
                        MaxValues[i],
                        loc=distribution_properties.loc[loc, "loc"],
                        scale=distribution_properties.loc[loc, "scale"],
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

            with open(save_to + "/" + fname, "w") as File:
                # write the first lines
                for i in range(len(SpatialRef)):
                    File.write(str(SpatialRef[i].decode()[:-2]) + "\n")

                for i in range(np.shape(RetunPeriodMap)[0]):
                    File.writelines(list(map(self.StringSpace, RetunPeriodMap[i, :])))
                    File.write("\n")

            # zip the file
            with zipfile.ZipFile(
                save_to + "/" + fname[:-4] + ".zip", "w", zipfile.ZIP_DEFLATED
            ) as newzip:
                newzip.write(save_to + "/" + fname, arcname=fname)
            # delete the file
            os.remove(save_to + "/" + fname)

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
        #    ToSave.to_csv(save_path + TraceFile,header = None, index = None)

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
