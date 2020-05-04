# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:51:02 2020

@author: mofarrag
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import gdal
import zipfile
from statsmodels import api as sm


class Inputs():

    def __init__(self, Name, Version=2 ):
        self.Name = Name
        self.Version = Version


    def ExtractHydrologicalInputs(self, WeatherGenerator, FilePrefix, realization,
                                   path, SWIMNodes, SavePath):
        """


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
                assert int(SWIMResultFile[-5:-4]) == int(SavePath[-1]), " Wrong files sync "
            else:
                assert int(SWIMResultFile[-6:-4]) == int(SavePath[-2:]), " Wrong files sync "
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
        SWIMData = pd.read_csv(path + "/" + SWIMResultFile, delimiter = r'\s+', header = None)
        Nodes = pd.read_csv(path + "/" + SWIMNodes, header = None)

        for i in range(len(Nodes)):
            SWIMData.loc[:,i + (ignoreColumns)].to_csv(SavePath + "/" + str(Nodes.loc[i,0]) + ".txt",
                        header = None, index = None)

    def StatisticalProperties(self, PathNodes, PathTS, StartDate, WarmUpPeriod, SavePlots, SavePath,
                              SeparateFiles = False, Filter = False, RIMResults = False):
        """
        =============================================================================
          StatisticalProperties(PathNodes, PathTS, StartDate, WarmUpPeriod, SavePlots, saveto)
        =============================================================================

        StatisticalProperties method reads the SWIM output file (.dat file) that
        contains the time series of discharge for some computational nodes
        and calculate some statistical properties

        the code assumes that the time series are of a daily temporal resolution, and
        that the hydrological year is 1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
            1-PathNodes : [String]
                the name of the file which contains the ID of the computational
                nodes you want to do the statistical analysis for, the ObservedFile
                should contain the discharge time series of these nodes in order.
            2-PathTS : [String]
                the name of the SWIM result file (the .dat file).
            3-StartDate : [string]
                the begining date of the time series.
            4-WarmUpPeriod : [integer]
                the number of days you want to neglect at the begining of the
                Simulation (warm up period).
            5-SavePlots : [Bool]
                DESCRIPTION.
            6-SavePath : [String]
                the path where you want to  save the statistical properties.
            7-SeparateFiles: [Bool]
                if the discharge data are stored in separate files not all in one file
                SeparateFiles should be True, default [False].
            8-Filter: [Bool]
                for observed or RIMresult data it has gaps of times where the
                model did not run or gaps in the observed data if these gap days
                are filled with a specific value and you want to ignore it here
                give Filter = Value you want

        Returns
        -------
            1-Statistical Properties.csv:
                file containing some statistical properties like mean, std, min, 5%, 25%,
                median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
                q100, q200, q500.
        """

        ComputationalNodes = np.loadtxt(PathNodes, dtype=np.uint16)
        # hydrographs
        if SeparateFiles:
            ObservedTS = pd.DataFrame()
            if RIMResults:
                for i in range(len(ComputationalNodes)):
                    ObservedTS.loc[:,int(ComputationalNodes[i])] =  self.ReadRIMResult(PathTS + "/" +
                                  str(int(ComputationalNodes[i])) + '.txt')
            else:
                for i in range(len(ComputationalNodes)):
                    ObservedTS.loc[:,int(ComputationalNodes[i])] = np.loadtxt(PathTS + "/" +
                              str(int(ComputationalNodes[i])) + '.txt') #,skiprows = 0

            StartDate = dt.datetime.strptime(StartDate,"%Y-%m-%d")
            EndDate = StartDate + dt.timedelta(days = ObservedTS.shape[0]-1)
            ind = pd.date_range(StartDate,EndDate)
            ObservedTS.index = ind
        else:
            ObservedTS = pd.read_csv(PathTS , delimiter = r'\s+', header = None)
            StartDate = dt.datetime.strptime(StartDate,"%Y-%m-%d")
            EndDate = StartDate + dt.timedelta(days = ObservedTS.shape[0]-1)
            ObservedTS.index = pd.date_range(StartDate,EndDate, freq="D")
            # delete the first two columns
            del ObservedTS[0], ObservedTS[1]
            ObservedTS.columns = ComputationalNodes

        # neglect the first year (warmup year) in the time series
        ObservedTS = ObservedTS.loc[StartDate + dt.timedelta(days = WarmUpPeriod):EndDate,:]

        # List of the table output, including some general data and the return periods.
        col_csv = ['mean', 'std', 'min', '5%', '25%', 'median',
                   '75%', '95%', 'max', 't_beg', 't_end', 'nyr']
        rp_name = ['q1.5', 'q2', 'q5', 'q10', 'q25', 'q50', 'q100', 'q200', 'q500', 'q1000']
        col_csv = col_csv + rp_name

        # In a table where duplicates are removed (np.unique), find the number of
        # gauges contained in the .csv file.
        # no_gauge = len(ComputationalNodes)
        # Declare a dataframe for the output file, with as index the gaugne numbers
        # and as columns all the output names.
        StatisticalPr = pd.DataFrame(np.nan, index=ComputationalNodes,
                                 columns=col_csv)
        StatisticalPr.index.name = 'ID'
        DistributionPr = pd.DataFrame(np.nan, index=ComputationalNodes,
                                 columns=['loc','scale'])
        DistributionPr.index.name = 'ID'
        # required return periods
        T = [1.5, 2, 5, 10, 25, 50, 50, 100, 200, 500, 1000]
        T = np.array(T)
        # these values are the Non Exceedance probability (F) of the chosen
        # return periods F = 1 - (1/T)
        # Non Exceedance propabilities
        #F = [1/3, 0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.995, 0.998]
        F = 1-(1/T)
        # Iteration over all the gauge numbers.
        for i in ComputationalNodes:
            QTS = ObservedTS.loc[:,i]
            # The time series is resampled to the annual maxima, and turned into a
            # numpy array.
            # The hydrological year is 1-Nov/31-Oct (from Petrow and Merz, 2009, JoH).
            amax = QTS.resample('A-OCT').max().values
            if type(Filter) != bool:
                amax = amax[amax != Filter]
            # A gumbel distribution is fitted to the annual maxima
            param_dist = gumbel_r.fit(amax)
            DistributionPr.loc[i,'loc'] = param_dist[0]
            DistributionPr.loc[i,'scale'] =param_dist[1]
            # Return periods from the fitted distribution are stored.
            # get the Discharge coresponding to the return periods
            Qrp = gumbel_r.ppf(F,loc=param_dist[0], scale=param_dist[1])
            # to get the Non Exceedance probability for a specific Value
            #gumbel_r.cdf(Qrp, loc=param_dist[0], scale=param_dist[1])
            # then calculate the the T (return period) T = 1/(1-F)

            # Plot the histogram and the fitted distribution, save it for each gauge.
            Qx = np.linspace(0, 1.5*float(amax.max()), 10000)
            pdf_fitted = gumbel_r.pdf(Qx, loc=param_dist[0], scale=param_dist[1])
            if SavePlots :
                plt.plot(Qx, pdf_fitted, 'r-')
                plt.hist(amax, density=True)
                plt.savefig(SavePath + "/" + "Figures/" + str(i) + '.png', format='png')
                plt.close()

            StatisticalPr.loc[i, 'mean'] = QTS.mean()
            StatisticalPr.loc[i, 'std'] = QTS.std()
            StatisticalPr.loc[i, 'min'] = QTS.min()
            StatisticalPr.loc[i, '5%'] = QTS.quantile(0.05)
            StatisticalPr.loc[i, '25%'] = QTS.quantile(0.25)
            StatisticalPr.loc[i, 'median'] = QTS.quantile(0.50)
            StatisticalPr.loc[i, '75%'] = QTS.quantile(0.75)
            StatisticalPr.loc[i, '95%'] = QTS.quantile(0.95)
            StatisticalPr.loc[i, 'max'] = QTS.max()
            StatisticalPr.loc[i, 't_beg'] = QTS.index.min()
            StatisticalPr.loc[i, 't_end'] = QTS.index.max()
            StatisticalPr.loc[i, 'nyr'] = (StatisticalPr.loc[i, 't_end'] -
                                     StatisticalPr.loc[i, 't_beg']).days / 365.25
            for irp, irp_name in zip(Qrp, rp_name):
                StatisticalPr.loc[i, irp_name] = irp

            # Print for prompt and check progress.
            print("Gauge", i, "done.")
        #
        # Output file
        StatisticalPr.to_csv(SavePath + "/" + "Statistical Properties.csv")
        self.StatisticalPr = StatisticalPr
        DistributionPr.to_csv(SavePath + "/" + "DistributionProperties.csv")
        self.DistributionPr = DistributionPr


    def WriteHQFile(self, NoNodes, StatisticalPropertiesFile, SaveTo):
        # Create a table of all nodes and sub-basins
        HQ = pd.DataFrame(index = list(range(NoNodes)), columns =['ID','2yrs','10yrs','100yrs'])
        HQ.loc[:,['2yrs','10yrs','100yrs']] = -1
        HQ.loc[:,'ID'] = range(1,NoNodes+1)
        StatisticalPr= pd.read_csv(StatisticalPropertiesFile)
        for i in range(len(StatisticalPr)):
        #i=0
            HQ.loc[StatisticalPr.loc[i,'ID']-1,'2yrs'] =  StatisticalPr.loc[i,'q2']
            HQ.loc[StatisticalPr.loc[i,'ID']-1,'10yrs'] = StatisticalPr.loc[i,'q10']
            HQ.loc[StatisticalPr.loc[i,'ID']-1,'100yrs'] = StatisticalPr.loc[i,'q100']

        # save the HQ file
        HQ.to_csv(SaveTo,float_format="%%.2f".format,index=False, header=None)


    @staticmethod
    def StringSpace(Inp):
            return str(Inp) + "  "

    def ReturnPeriod(self,MapsPath, prefix, DistributionPrF, TraceF, SubsF,
                     replacementF, HydrologicalInputsPath, SubIDMapF, ExtraSubsF,
                     Fromfile, Tofile, SaveTo, wpath):

        AllResults=os.listdir(MapsPath)
        # list of the Max Depth files only
        MaxDepthList = list()
        for i in range(len(AllResults)):
            if AllResults[i].startswith(prefix):
                MaxDepthList.append(AllResults[i])
        # Read Inputs
        # read the Distribution parameters for each upstream computatiopnal node
        DistributionPr = pd.read_csv(DistributionPrF)
        USnode = pd.read_csv(TraceF, header = None)
        USnode.columns = ['SubID','US','DS']
        # get the sub basin Id from the guide file it is the same shape in RIM1.0 and RIM2.0
        SubsID = pd.read_csv(SubsF, header = None, usecols=[0])

        ReplacementSub = pd.read_csv(replacementF)

        # read the hydrograph for all the US nodes
        #StartDate = "1950-1-1"
        #StartDate = dt.datetime.strptime(StartDate,"%Y-%m-%d")
        #ind = pd.date_range(StartDate, StartDate + dt.timedelta(days = NoYears*365), freq = "D")

        ind = range(1,len(pd.read_csv(HydrologicalInputsPath + "/"+str(int(USnode.loc[SubsID.loc[10,0] - 1,'US']))+".txt").values))

        Hydrographs = pd.DataFrame(index = ind, columns = SubsID[0].to_list())

        for i in range(len(SubsID)):
        #    i=1
            # search for the SubId in the USnode or it is listed by order so subID=343 exist
            # in the row 342 (SubID-1)
            # np.where(USnode['SubID'] == SubsID.loc[i,0])
            try:
                if int(USnode.loc[SubsID.loc[i,0] - 1,'US']) != -1 :
                    Hydrographs.loc[:,SubsID.loc[i,0]] = pd.read_csv(HydrologicalInputsPath + "/"+str(int(USnode.loc[SubsID.loc[i,0] - 1,'US']))+".txt").values[:len(Hydrographs)]
            except :
                OtherSubLoc = np.where(ReplacementSub['missing'] == SubsID.loc[i,0])[0][0]
                if int(USnode.loc[ReplacementSub.loc[OtherSubLoc,'replacement'] - 1,'US']) != -1 :
                   Hydrographs.loc[:,SubsID.loc[i,0]] = pd.read_csv(HydrologicalInputsPath + "/"+str(int(USnode.loc[ReplacementSub.loc[OtherSubLoc,'replacement'] - 1,'US']))+".txt").values[:len(Hydrographs)]

        # read sub basin map id
        SubIDMap = gdal.Open(SubIDMapF)
        SubIDMapV = SubIDMap.ReadAsArray()
        #NoValue = SubIDMap.GetRasterBand(1).GetNoDataValue()
        #SubIDMapV[SubIDMapV == NoValue] = 0
        #plt.imshow(SubIDMapV)

        # read the added subs reference text file
        ExtraSubs = pd.read_csv(ExtraSubsF)

        # function to write the numbers in the ASCII file

        #read Max depth map
        check = list()
        Klist = list()

        if Tofile == "end" or Tofile > len(MaxDepthList):
            Tofile = len(MaxDepthList)


        #Fromfile = 48
        #Tofile = Fromfile +1

        for k in range(Fromfile,Tofile):


            try:
                # open the zip file
                Compressedfile = zipfile.ZipFile(MapsPath + "/" + MaxDepthList[k])
            except:
                print("Error Opening the compressed file")
                check.append(MaxDepthList[k][len(prefix):-4])
                Klist.append(k)
                continue

            # get the file name
            fname = Compressedfile.infolist()[0]
            # get the time step from the file name
            timestep = int(fname.filename[len(prefix):-4])
            print("File= " + str(timestep))

            ASCIIF = Compressedfile.open(fname)
            f = ASCIIF.readlines()
            SpatialRef = f[:6]
            ASCIIRaw = f[6:]
            # ASCIIF = Compressedfile.open(fname)
            # ASCIIRaw = ASCIIF.readlines()[6:]
            rows = len(ASCIIRaw)
            cols = len(ASCIIRaw[0].split())
            MaxDepth = np.ones((rows,cols), dtype = np.float32)
            # read the ascii file
            for i in range(rows):
                x = ASCIIRaw[i].split()
                MaxDepth[i,:] = list(map(float, x ))

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
            MaxValuedf = Hydrographs.loc[timestep-14:timestep,:]
            MaxValues = MaxValuedf.max().values.tolist()
            T = list()

            # Calculate the the Return period for the max Q at this time step for each
            for i in range(len(MaxValues)):
                # if the sub basin is a lateral and not routed in RIM it will not have a
                # hydrograph
                if np.isnan(MaxValues[i]):
                    T.append(np.nan)
                if not np.isnan(MaxValues[i]):
                    #np.where(USnode['SubID'] == SubsID.loc[i,0])
                    try:
                        DSnode = USnode.loc[SubsID.loc[i,0]-1,'US']
                        loc = np.where(DistributionPr['ID'] == DSnode)[0][0]
                    except IndexError:
                        OtherSubLoc = np.where(ReplacementSub['missing'] == SubsID.loc[i,0])[0][0]
                        DSnode = USnode.loc[ReplacementSub.loc[OtherSubLoc,'replacement']-1,'US']
                        loc = np.where(DistributionPr['ID'] == DSnode)[0][0]

                    # to get the Non Exceedance probability for a specific Value
                    F = gumbel_r.cdf(MaxValues[i], loc=DistributionPr.loc[loc,'loc'],
                                 scale=DistributionPr.loc[loc,'scale'])
                    # then calculate the the T (return period) T = 1/(1-F)
                    T.append(round(1/(1-F),2))

            try:
                RetunPeriodMap = np.ones((rows,cols), dtype = np.float32) * 0
                for i in range(rows):
                    for j in range(cols):
                        # print("i = " + str(i) + ", j= " + str(j))
                        if not np.isnan(MaxDepth[i,j]):
                            if MaxDepth[i,j] > 0 :
                                # print("i = " + str(i) + ", j= " + str(j))
                                # if the sub basin is in the Sub ID list
                                if SubIDMapV[i,j] in SubsID[0].tolist():
                                    # print("Sub = " + str(SubIDMapV[i,j]))
                                    # go get the return period directly
                                    RetunPeriodMap[i,j] = T[np.where(SubsID[0] == SubIDMapV[i,j])[0][0]]
                                else :
                                    # print("Extra  Sub = " + str(SubIDMapV[i,j]))
                                    # the sub ID is one of the added subs not routed by RIM
                                    # so it existed in the ExtraSubs list with a reference to
                                    # a SubID routed by RIM
                                    RIMSub  = ExtraSubs.loc[np.where(ExtraSubs['addSub'] == SubIDMapV[i,j])[0][0],'RIMSub']
                                    RetunPeriodMap[i,j] = T[np.where(SubsID[0] == RIMSub)[0][0]]
            except :
                print("Error")
                check.append(timestep)
                Klist.append(k)
                continue

            # save the return period ASCII file
            fname = "ReturnPeriod" +  str(timestep) + ".asc"

            with open(SaveTo + "/" + fname,'w') as File:
                # write the first lines
                for i in range(len(SpatialRef)):
                    File.write(str(SpatialRef[i].decode()[:-2]) + "\n")

                for i in range(np.shape(RetunPeriodMap)[0]):
                    File.writelines(list(map(self.StringSpace,RetunPeriodMap[i,:])))
                    File.write("\n")

            # zip the file
            with zipfile.ZipFile(SaveTo + "/" + fname[:-4] + ".zip","w",zipfile.ZIP_DEFLATED) as newzip:
                newzip.write(SaveTo + "/" + fname, arcname = fname)
            # delete the file
            os.remove(SaveTo + "/" + fname)

        check = list(zip(check,Klist))
        if len(check) > 0:
            np.savetxt(wpath + "CheckWaterDepth.txt",check,fmt='%6d')


    @staticmethod
    def ReadRIMResult(Path):
        """
        =======================================
           ReadRIMResult(Path)
        =======================================

        Parameters
        ----------
            Path : [String]
                path to the RIM output file you want to read, the file should
                contain two columns the first is the index of the day and the
                second is the discharge value, the method fills missed days
                with zeros

        Returns
        -------
            f2 : TYPE
                DESCRIPTION.

        """

        f = np.loadtxt(Path, delimiter = ",")

        f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
        f2 = list()
        for j in range(len(f1)):
            # if the index exist in the original list
            if f1[j] in f[:,0]:
                # put the coresponding value in f2
                f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
            else:
                # if it does not exist put zero
                f2.append(0)
        return f2


    def CreateTraceALL(self, ConfigFilePath, RIMSubsFilePath, TraceFile, USonly=1,
                       HydrologicalInputsFile=''):

        # reading the file
        Config = pd.read_csv(ConfigFilePath, header = None)
        # process the Configuration file
        # get the Route rows from the file
        Route = pd.DataFrame(columns = ['No','DSnode','SubID','USnode','No2'])

        j=0
        for i in range(len(Config)):
            if Config[0][i].split()[0] == 'route':
                Route.loc[j,:] = list(map(int,Config[0][i].split()[1:]))
                j = j + 1

        # get RIM Sub-basins
        Subs = pd.read_csv(RIMSubsFilePath, header = None)
        Subs = Subs.rename(columns = {0:"SubID"})

        if self.Version == 1:
            Subs['US'] = -1
            Subs['DS'] = -1
        else:
            Subs['US'] = None
            Subs['DS'] = None

        for i in range(len(Subs)):
            try:
                # if the sub-basin is in the route array so it is routed by SWIM
                loc = np.where(Route['SubID'] == Subs.loc[i,'SubID'])[0][0]
                Subs.loc[i,'US'] = int(Route.loc[loc,'USnode'])
                Subs.loc[i,'DS'] = int(Route.loc[loc,'DSnode'])
            except IndexError:
                # if the sub-basin is not in the route array so it is not routed by SWIM
                # but still can be routed using RIM
                if self.Version == 1:
                    Subs.loc[i,'US'] = -1
                    Subs.loc[i,'DS'] = -1
                else:
                    # Subs.loc[i,'US'] = None
                    # Subs.loc[i,'DS'] = None
                    Subs.loc[i,'US'] = -1
                    Subs.loc[i,'DS'] = -1

        # Save the file with the same format required for the hg R code
        if self.Version == 1:
        #    ToSave = Subs.loc[:,['US','SubID']]
        #    ToSave['Extra column 1'] = -1
        #    ToSave['Extra column 2'] = -1
            Subs.to_csv(TraceFile,header = True, index = None)

            onlyRouted = Subs[Subs['US'] != -1][Subs['DS'] != -1]

            if USonly == 1:
                All = onlyRouted['US'].tolist()
            else:
                All = onlyRouted['US'].tolist() + onlyRouted['DS'].tolist()
            np.savetxt(HydrologicalInputsFile , All, fmt = '%d')

        else:
            Subs.to_csv(TraceFile, index = None, header = True)
        #    ToSave = Subs.loc[:,['SubID','US']]
        #    ToSave['Extra column 1'] = -1
        #    ToSave['Extra column 2'] = -1
        #    ToSave.to_csv(SavePath + TraceFile,header = None, index = None)



class CrossSections():
    def __init__(self,name):
        self.name = name


    def reg_plot(self,x , y, xlab, ylab, xlgd, ylgd, title, filename,
                 log, logandlinear, seelim, Save = False, *args, **kwargs):
        """
        =============================================================================
        reg_plot(x , y, minmax_XS_area, xlab, ylab, xlgd, ylgd, title, filename,
                 log, logandlinear, seelim, Save = False, *args, **kwargs)
        =============================================================================

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
                xmin = kwargs['XLim'][0]
                xmax = kwargs['XLim'][1]

    #    if log is True:
    #        # Linear regression with the ordinaty least squares method (Y, X).
    #        # sm.add_constant is required to get the intercept.
    #        # NaN values are dropped.
    #        results = sm.OLS(np.log10(y), sm.add_constant(np.log10(x)),
    #                         missing='drop').fit()
    #    elif log is False:
    #        results = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
        # fit the relation between log(x) and log(y)
        results_log = sm.OLS(np.log10(y), sm.add_constant(np.log10(x)),
                             missing='drop').fit()
        # fit the relation between x and y
        results_lin = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
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
        coeff = 10**results_log.params[0]
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
            results_log_df.index = ['intercept_' + filename, 'slope_' + filename]
            if Save:
                results_log_df.to_csv(filename + '_powerlaw.csv', sep=',')
            # linear data
            results_lin_df = self.results_summary_to_dataframe(results_lin)
            results_lin_df.index = ['intercept_' + filename, 'slope_' + filename]
            if Save:
                results_lin_df.to_csv(filename + '_lin.csv', sep=',')


        if logandlinear is False:
            # Plot the points and the regression line
            plt.scatter(x, y)
            x_plot = np.linspace(x.min(), x.max(), 1000)

            # Plot for log-log regression
            if log is True:
                plt.plot(x_plot, coeff*(x_plot**exp))
                plt.xscale('log')
                plt.yscale('log')
                plt.annotate('$%s = %.4f%s^{%.4f}$' % (ylgd, coeff, xlgd, exp) +
                             '\n' + '$R^2 = %.4f$' % rsq_log,
                             xy=(0.05, 0.90), xycoords='axes fraction')
                plt.xlim(0.5*x.min(), 1.5*x.max())
                plt.ylim(0.5*y.min(), 1.5*y.max())
            # Plot for linear regression
            elif log is False:
                plt.plot(x_plot, intercept + slope*x_plot)
                plt.annotate('%s = %.4f + %.4f%s' % (ylgd, intercept, slope, xlgd) +
                             '\n' + '$R^2 = %.4f$' % rsq_lin,
                             xy=(0.05, 0.90), xycoords='axes fraction')
                plt.xlim(0,
                         x.max() + 0.1*(x.max() - x.min()))
                plt.ylim(0,
                         y.max() + 0.1*(y.max() - y.min()))

            plt.xlabel(xlab)
            plt.ylabel(ylab)
            # vertical lines for min and max cross section area
            if seelim is True:
                plt.axvline(x=xmin, color='red',
                            ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color='red',
                            ymin=0.25, ymax=0.75)
            plt.title(title)

            if Save:
                plt.savefig(filename + '.png', dpi=400)
                plt.close()
            if log is True:
                sm.graphics.plot_regress_exog(results_log, 1)
                if Save:
                    plt.savefig(filename + '_powerlaw_resid.png', dpi=400)
            elif log is False:
                sm.graphics.plot_regress_exog(results_lin, 1)
                if Save:
                    plt.savefig(filename + '_lin_resid.png', dpi=400)
                    plt.close()

        # Plot the power law and linear regressions on a log-log scale
        # and on a linear scale to compare the fits
        elif logandlinear is True:
            # Plot on a log-log scale
            plt.scatter(x, y, c='grey', marker="x")
            x_plot = np.linspace(x.min(), x.max(), 1000)
            plt.plot(x_plot, coeff*(x_plot**exp))
            plt.xscale('log')
            plt.yscale('log')
            plt.annotate('$%s = %.4f%s^{%.4f}$' % (ylgd, coeff, xlgd, exp) +
                         '\n' + '$R^2 = %.4f$' % rsq_log,
                         xy=(0.05, 0.90), xycoords='axes fraction')
            plt.plot(x_plot, intercept + slope*x_plot)
            plt.annotate('$%s = %.4f + %.4f%s$' % (ylgd, intercept, slope, xlgd) +
                         '\n' + '$R^2 = %.4f$' % rsq_lin,
                         xy=(0.05, 0.80), xycoords='axes fraction')
            plt.xlim(0.5*x.min(), 1.5*x.max())
            plt.ylim(0.5*y.min(), 1.5*y.max())
            plt.xlabel(xlab)
            plt.ylabel(ylab)

            if seelim is True:
                plt.axvline(x=xmin, color='red',
                            ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color='red',
                            ymin=0.25, ymax=0.75)
            plt.title(title)
            if Save:
                plt.savefig(filename + '_logscale.png', dpi=400)
                plt.close()
            # Plot on a linear scale
            plt.scatter(x, y, c='grey', marker="x")
            x_plot = np.linspace(x.min(), x.max(), 1000)
            plt.plot(x_plot, coeff*(x_plot**exp))
            plt.annotate('$%s = %.4f%s^{%.4f}$' % (ylgd, coeff, xlgd, exp) +
                         '\n' + '$R^2 = %.4f$' % rsq_log,
                         xy=(0.05, 0.90), xycoords='axes fraction')
            plt.plot(x_plot, intercept + slope*x_plot)
            plt.annotate('$%s = %.4f + %.4f%s$' % (ylgd, intercept, slope, xlgd) +
                         '\n' + '$R^2 = %.4f$' % rsq_lin,
                         xy=(0.05, 0.80), xycoords='axes fraction')
            plt.xlim(0,
                     x.max() + 0.1*(x.max() - x.min()))
            plt.ylim(0,
                     y.max() + 0.1*(y.max() - y.min()))
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            if seelim is True:
                plt.axvline(x=xmin, color='red',
                            ymin=0.25, ymax=0.75)
                plt.axvline(x=xmax, color='red',
                            ymin=0.25, ymax=0.75)
            plt.title(title)
            if Save:
                plt.savefig(filename + '_linscale.png', dpi=400)
                plt.close()


    def reg_plot_river(self, river_lst, data, minmax_XS_area, filename, log,
                       Save = False):
        """
        Use the ordinary least squares to make a regression and plot the
        output.

        This version makes use of the field 'river' in data to define a subset
        of gauges used for the regression.
        """
        df_output = pd.DataFrame()
        fig1 = plt.figure(figsize=(11.69, 16.53))
        for i, riveri in enumerate(river_lst):
            fig1.add_subplot(5, 2, i+1)
            # The subset is defined checking the filed 'river' and using the
            # river names provided in river_lst
            datai = data[data['river'].str.contains(riveri, case=False) == True]
            # Warning, this df is passed outside of the function.
            min_XSUSarea = minmax_XS_area.at[riveri, 'Min_UpXSArea']
            max_XSUSarea = minmax_XS_area.at[riveri, 'Max_UpXSArea']
            print('There are {} gauges related to {}'.format(datai.shape[0], riveri))
            # The regression is done if there are at least 4 gauges
            if datai.shape[0] > 3:
                x = datai['Area_Final']
                y = datai['q2']
                plt.scatter(x, y)
                X_plot = np.linspace(x.min(), x.max(), 1000)
                plt.ylabel('$HQ_2 [m^3/s]$')
                plt.xlabel('$Drainage Area [km^2]$')
                plt.axvline(x=min_XSUSarea, color='red', ymin=0.25, ymax=0.75)
                plt.axvline(x=max_XSUSarea, color='red', ymin=0.25, ymax=0.75)
                # Log-Log regression
                if log is True:
                    resultslog = sm.OLS(np.log10(y),
                                sm.add_constant(np.log10(x)), # Required to get the interc.
                                missing='drop').fit() # NaN values are dropped.
                    print(resultslog.summary())
                    results_df = self.results_summary_to_dataframe(resultslog)
                    results_df.index = ['interceptlog_' + riveri,
                                            'slopelog_' + riveri]
                    df_output = df_output.append(results_df)
                    coeffi = 10**resultslog.params[0]
                    expi = resultslog.params[1]
                    rsqi = resultslog.rsquared
                    plt.plot(X_plot,
                             (10**resultslog.params[0])*(X_plot**resultslog.params[1]))
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.annotate('$HQ_2 = %.3fA^{%.3f}$' % (coeffi, expi) +
                                 '\n' + '$R^2 = %.3f$' % rsqi,
                                 xy=(0.05, 0.85), xycoords='axes fraction')
                    plt.xlim(0.5*x.min(), 1.5*x.max())
                    plt.ylim(0.5*y.min(), 1.5*y.max())
                    plt.title(filename + ' ' + riveri + ' log-log')
                # Linear regression
                elif log is False:
                    results = sm.OLS(y,
                                     sm.add_constant(x), # Required to get the interc.
                                     missing='drop').fit() # NaN values are dropped.
                    print(results.summary())
                    results_df = self.results_summary_to_dataframe(results)
                    results_df.index = ['interceptlinear_' + riveri,
                                            'slopelinear_' + riveri]
                    df_output = df_output.append(results_df)
                    intercept = results.params[0]
                    slope = results.params[1]
                    rsq = results.rsquared
                    plt.plot(X_plot, intercept + slope*X_plot)
                    plt.annotate('Q2 = %.4f + %.4fA' % (intercept, slope) +
                                 '\n' + '$R^2 = %.4f$' % rsq,
                                 xy=(0.05, 0.85), xycoords='axes fraction')
                    plt.xlim(0, 1.1*x.max())
                    plt.ylim(0, 1.1*y.max())
                    plt.title(filename + ' ' + riveri + ' linear')
                #plt.show()
        plt.tight_layout()
        # Plot the residuals.
        # Export for log-log regression
        if log is True:
            if Save:
                fig1.savefig(filename + '_LogLog.png', dpi=400)
            fig2 = sm.graphics.plot_regress_exog(resultslog, 1)
            if Save:
                fig2.savefig(filename + '_ResidLogLog.png', dpi=400)
                df_output.to_csv(filename + '_log.csv', sep=',')
        # Export for linear regression
        elif log is False:
            if Save:
                fig1.savefig(filename + '_Linear.png', dpi=400)
            fig2 = sm.graphics.plot_regress_exog(results, 1)
            if Save:
                fig2.savefig(filename + '_ResidLinear.png', dpi=400)
                df_output.to_csv(filename + '_linear.csv', sep=',')
        if Save:
            plt.close(fig1)
            plt.close(fig2)


    def reg_plot_subbasin(subbasin_lst, data, filename, log, redfact):
        """
        Use the ordinary least squares to make a regression and plot the
        output.

        This version makes use of the field 'Subbasin' in data to define a subset
        of gauges used for the regression.

        redfact is a real between 0 and 1, and is used to define for each
        subbasin an area threshold that will limit the number of small gauges
        selected.
        """
        df_output = pd.DataFrame()
        fig1 = plt.figure(figsize=(11.69, 16.53))
        for i, subbasini in enumerate(subbasin_lst):
            fig1.add_subplot(5, 2, i+1)
            datai = data[data['Subbasin'].str.contains(subbasini, case=False) == True]
            min_XSUSarea = minmax_XS_area.at[subbasini, 'Min_UpXSArea']
            min_gauge_area = redfact*min_XSUSarea
            max_XSUSarea = minmax_XS_area.at[subbasini, 'Max_UpXSArea']
            datai_red = datai[datai['Area_Final'] > min_gauge_area]
            print('There are {} gauges used for {}'.format(datai_red.shape[0],
                                                             subbasini))
            if datai.shape[0] > 3:
                x = datai_red['Area_Final']
                y = datai_red['q2']
                plt.scatter(x, y)
                X_plot = np.linspace(x.min(), x.max(), 1000)

                if log is True:
                    resultslog = sm.OLS(np.log10(y),
                                sm.add_constant(np.log10(x)), # Required to get the interc.
                                missing='drop').fit() # NaN values are dropped.
                    print(resultslog.summary())
                    results_df = self.results_summary_to_dataframe(resultslog)
                    results_df.index = ['interceptlog_' + subbasini,
                                            'slopelog_' + subbasini]
                    df_output = df_output.append(results_df)
                    coeff = 10**resultslog.params[0]
                    exp = resultslog.params[1]
                    rsq = resultslog.rsquared
                    plt.plot(X_plot,
                             (10**resultslog.params[0])*(X_plot**resultslog.params[1]))
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.annotate('$Q2 = %.3fA^{%.3f}$' % (coeff, exp) +
                                 '\n' + '$R^2 = %.3f$' % rsq,
                                 xy=(0.05, 0.85), xycoords='axes fraction')
                    #plt.xlim(0.5*x.min(), 1.5*x.max())
                    plt.xlim(0.5*min_gauge_area, 1.5*max_XSUSarea)
                    plt.ylim(0.5*y.min(), 1.5*y.max())
                    plt.title(filename + ' ' + subbasini + ' log-log')
                elif log is False:
                    results = sm.OLS(y,
                                     sm.add_constant(x), # Required to get the interc.
                                     missing='drop').fit() # NaN values are dropped.
                    print(results.summary())
                    results_df = self.results_summary_to_dataframe(results)
                    results_df.index = ['interceptlinear_' + subbasini,
                                            'slopelinear_' + subbasini]
                    df_output = df_output.append(results_df)
                    intercept = results.params[0]
                    slope = results.params[1]
                    rsq = results.rsquared
                    plt.plot(X_plot, intercept + slope*X_plot)
                    plt.annotate('Q2 = %.4f + %.4fA' % (intercept, slope) +
                                 '\n' + '$R^2 = %.4f$' % rsq,
                                 xy=(0.05, 0.85), xycoords='axes fraction')
                    # plt.xlim(0.8*x.min(), 1.1*x.max())
                    plt.xlim(0, 1.1*max_XSUSarea)
                    plt.ylim(0, 1.1*y.max())
                    plt.title(filename + ' ' + subbasini + ' linear')

                plt.ylabel('Q2 [m3/s]')
                plt.xlabel('Drainage Area [km2]')
                plt.axvline(x=min_XSUSarea, color='red', ymin=0.25, ymax=0.75)
                plt.axvline(x=max_XSUSarea, color='red', ymin=0.25, ymax=0.75)
                # plt.show()
        plt.tight_layout()
        if log is True:
            fig1.savefig(filename + '_LogLog.png', dpi=400)
            fig2 = sm.graphics.plot_regress_exog(resultslog, 1)
            fig2.savefig(filename + '_ResidLogLog.png', dpi=400)
            df_output.to_csv(filename + '_LogLog.csv', sep=',')
        elif log is False:
            fig1.savefig(filename + '_Linear.png', dpi=400)
            fig2 = sm.graphics.plot_regress_exog(results, 1)
            fig2.savefig(filename + '_ResidLinear.png', dpi=400)
            df_output.to_csv(filename + '_Linear.csv', sep=',')
        plt.close(fig1)
        plt.close(fig2)

    @staticmethod
    def results_summary_to_dataframe(res):
        """
        Take the result of a statsmodel results table and transforms
        it into a dataframe
        """

        pvals = res.pvalues
        coeff = res.params
        conf_lower = res.conf_int()[0]
        conf_higher = res.conf_int()[1]
        stderr = res.bse
        tvals = res.tvalues
        rsq = res.rsquared
        rsq_adj = res.rsquared_adj
        no_obs = res.nobs

        res_df = pd.DataFrame({"pvals": pvals,
                               "coeff": coeff,
                               "conf_lower": conf_lower,
                               "conf_higher": conf_higher,
                               "std_err": stderr,
                               "tvals": tvals,
                               "R-squared": rsq,
                               "R-squared_adj": rsq_adj,
                               "no_obs": no_obs
                               })

        # Reordering...
        res_df = res_df[["coeff", "std_err", "tvals",
                         "pvals", "conf_lower", "conf_higher", "R-squared",
                         "R-squared_adj", "no_obs"]]
        return res_df
