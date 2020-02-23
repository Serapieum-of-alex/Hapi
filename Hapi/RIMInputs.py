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

class Inputs():
    
    def __init__(self,name):
        self.name = name
    
    def StatisticalProperties(self, PathNodes, PathTS, StartDate, WarmUpPeriod, SavePlots, SavePath):
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

        Returns
        -------
            1-Statistical Properties.csv:
                file containing some statistical properties like mean, std, min, 5%, 25%,
                median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50, 
                q100, q200, q500.
        """
        
        ComputationalNodes = np.loadtxt(PathNodes, dtype=np.uint16)
        # hydrographs
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
        rp_name = ['q1.5', 'q2', 'q5', 'q10', 'q25', 'q50', 'q100', 'q200', 'q500']
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
        T = [1.5, 2, 5, 10, 25, 50, 50, 100, 200, 500]
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
                plt.hist(amax, normed=True)
                plt.savefig(SavePath + "Figures/" + str(i) + '.png', format='png')
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