"""Created on Wed May 16 03:50:00 2018.

@author: Mostafa
"""
import datetime as dt
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from osgeo import gdal
from pyramids.raster import Raster as raster
from rasterio.plot import show

# from datetime import datetime
# import geopandas as gpd
from rasterstats import zonal_stats

import Hapi


class Inputs:
    """Inputs.

    Inputs class contains methods to prepare the inputs for the distributed
    hydrological model

    Methods:
        1- prepareInputs
        2- extractParametersBoundaries
        3- extractParameters
        4- createLumpedInputs
        5- renameFiles
        6- changetext2time
        7- ReadExcelData
        8- ListAttributes
    """

    def __init__(self):
        """no parameters needed for instantiating the object."""
        pass

    def prepareInputs(src: str, input_folder: str, folder_name: str):
        """prepareInputs.

        this function prepare downloaded raster data to have the same align and
        nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
        and return a folder with the output rasters with a name "New_Rasters"

        Parameters
        ----------
        src: [str]
            path to the spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns) A_path should include the name of the raster
            and the extension like "data/dem.tif".
        input_folder: [str]
            path of the folder of the rasters you want to adjust their
            no of rows, columns and resolution (alignment) like raster A
            the folder should not have any other files except the rasters.
        folder_name: [str]
            name to create a folder to store resulted rasters.

        Example
        -------
        Ex1:
            >>> dem_path="01GIS/inputs/4000/acc4000.tif"
            >>> prec_in_path="02Precipitation/CHIRPS/Daily/"
            >>> Inputs.prepareInputs(dem_path,prec_in_path,"prec")
        Ex2:
            >>> dem_path="01GIS/inputs/4000/acc4000.tif"
            >>> outputpath="00inputs/meteodata/4000/"
            >>> evap_in_path="03Weather_Data/evap/"
            >>> Inputs.prepareInputs(dem_path,evap_in_path,outputpath+"evap")
        """
        if not isinstance(folder_name, str):
            print("folder_name input should be string type")
        # create a new folder for new created alligned rasters in temp
        # check if you can create the folder
        try:
            os.makedirs(os.path.join(os.environ["TEMP"], "AllignedRasters"))
        except WindowsError:
            # if not able to create the folder delete the folder with the same name and create one empty
            shutil.rmtree(os.path.join(os.environ["TEMP"] + "/AllignedRasters"))
            os.makedirs(os.path.join(os.environ["TEMP"], "AllignedRasters"))

        temp = os.environ["TEMP"] + "/AllignedRasters/"

        # match alignment
        print(
            "- First alligned files will be created in a folder 'AllignedRasters' in the Temp folder in you "
            "environment variable"
        )
        raster.matchDataAlignment(src, input_folder, temp)
        # create new folder in the current directory for alligned and nodatavalue matched cells
        try:
            os.makedirs(os.path.join(os.getcwd(), folder_name))
        except (WindowsError, FileExistsError):
            raise FileExistsError(
                f"please The function is trying to create a folder with a name {folder_name}  "
                f"to complete the process if there is a folder with the same name please rename "
                f"it to other name"
            )
        # match nodata value
        print(
            "- Second matching NoDataValue from the DEM raster too all raster will be created in the outputpath"
        )
        raster.cropAlignedFolder(temp, src, f"{folder_name}/")
        # delete the processing folder from temp
        shutil.rmtree(temp)

    @staticmethod
    def extractParametersBoundaries(Basin):
        """extractParametersBoundaries.

        extractParametersBoundaries

        Parameters
        ----------
        Basin : [Geodataframe]
            gepdataframe of catchment polygon, make sure that the geodataframe contains
            one row only, if not merge all the polygons in the shapefile first.

        Returns
        -------
        UB : [list]
            list of the upper bound of the parameters.
        LB : [list]
            list of the lower bound of the parameters.

        the parameters are
            ["tt", "sfcf","cfmax","cwh","cfr","fc","beta",
             "lp","k0","k1","k2","uzl","perc", "maxbas"]
        """
        ParametersPath = os.path.dirname(Hapi.__file__)
        ParametersPath = ParametersPath + "/parameters"
        ParamList = [
            "01_tt",
            "02_rfcf",
            "03_sfcf",
            "04_cfmax",
            "05_cwh",
            "06_cfr",
            "07_fc",
            "08_beta",
            "09_etf",
            "10_lp",
            "11_k0",
            "12_k1",
            "13_k2",
            "14_uzl",
            "15_perc",
            "16_maxbas",
            "17_K_muskingum",
            "18_x_muskingum",
        ]

        raster_obj = rasterio.open(ParametersPath + "/max/" + ParamList[0] + ".tif")
        Basin = Basin.to_crs(crs=raster_obj.crs)
        # max values
        UB = list()
        for i in range(len(ParamList)):
            raster_obj = rasterio.open(ParametersPath + "/max/" + ParamList[i] + ".tif")
            array = raster_obj.read(1)
            affine = raster_obj.transform
            UB.append(
                zonal_stats(Basin, array, affine=affine, stats=["max"])[0]["max"]
            )  # stats=['min', 'max', 'mean', 'median', 'majority']

        # min values
        LB = list()
        for i in range(len(ParamList)):
            raster_obj = rasterio.open(ParametersPath + "/min/" + ParamList[i] + ".tif")
            array = raster_obj.read(1)
            affine = raster_obj.transform
            LB.append(zonal_stats(Basin, array, affine=affine, stats=["min"])[0]["min"])

        Par = pd.DataFrame(index=ParamList)

        Par["UB"] = UB
        Par["LB"] = LB
        # plot the given basin with the parameters raster
        ax = show((raster_obj, 1), with_bounds=True)
        Basin.plot(facecolor="None", edgecolor="blue", linewidth=2, ax=ax)
        # ax.set_xbound([Basin.bounds.loc[0,'minx']-10,Basin.bounds.loc[0,'maxx']+10])
        # ax.set_ybound([Basin.bounds.loc[0,'miny']-1, Basin.bounds.loc[0,'maxy']+1])

        return Par

    def extractParameters(self, gdf, scenario, as_raster=False, save_to=""):
        """extractParameters.

        extractParameters method extracts the parameter rasters at the location
        of the source raster, there are 12 set of parameters 10 sets of parameters
        (Beck et al., (2016)) and the max, min and average of all sets


        Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles,
        T. R. M. & Jaap Schellekens, and L. A. B. (2016) Global-scale
        regionalization of hydrologic model parameters-Supporting materials
        3599–3622. doi:10.1002/2015WR018247.Received

        Parameters
        ----------
        gdf: [Geodataframe]
            gepdataframe of catchment polygon, make sure that the geodataframe contains
            one row only, if not merge all the polygons in the shapefile first.
        scenario: [str]
            name of the parameter set, there are 12 sets of parameters
            ["1","2","3","4","5","6","7","8","9","10","avg","max","min"]
        as_raster: [bool]
            Default is False.
        save_to: [str]
            path to a directory where you want to save the rasters.

        Returns
        -------
        Parameters : [list]
            list of the upper bound of the parameters.


        the parameters are
            ["tt", rfcf,"sfcf","cfmax","cwh","cfr","fc","beta",'etf'
             "lp","k0","k1","k2","uzl","perc", "maxbas",'K_muskingum',
             'x_muskingum']
        """
        ParametersPath = os.path.dirname(Hapi.__file__)
        ParametersPath = ParametersPath + "/parameters/" + scenario
        ParamList = [
            "01_tt",
            "02_rfcf",
            "03_sfcf",
            "04_cfmax",
            "05_cwh",
            "06_cfr",
            "07_fc",
            "08_beta",
            "09_etf",
            "10_lp",
            "11_k0",
            "12_k1",
            "13_k2",
            "14_uzl",
            "15_perc",
            "16_maxbas",
            "17_K_muskingum",
            "18_x_muskingum",
        ]

        if not as_raster:
            raster_obj = rasterio.open(f"{ParametersPath}/{ParamList[0]}.tif")
            gdf = gdf.to_crs(crs=raster_obj.crs)
            # max values
            Par = list()
            for i in range(len(ParamList)):
                raster_obj = rasterio.open(f"{ParametersPath}/{ParamList[i]}.tif")
                array = raster_obj.read(1)
                affine = raster_obj.transform
                Par.append(
                    zonal_stats(gdf, array, affine=affine, stats=["max"])[0]["max"]
                )  # stats=['min', 'max', 'mean', 'median', 'majority']

            # plot the given basin with the parameters raster

            # Plot DEM
            ax = show((raster_obj, 1), with_bounds=True)
            gdf.plot(facecolor="None", edgecolor="blue", linewidth=2, ax=ax)
            # ax.set_xbound([Basin.bounds.loc[0,'minx']-10,Basin.bounds.loc[0,'maxx']+10])
            # ax.set_ybound([Basin.bounds.loc[0,'miny']-1, Basin.bounds.loc[0,'maxy']+1])

            return Par
        else:
            Inputs.prepareInputs(gdf, f"{ParametersPath}/", save_to)

    @staticmethod
    def createLumpedInputs(Path):
        """createLumpedInputs.

        createLumpedInputs method generate a lumped parameters from
        distributed parameters by taking the average
        Parameters
        ----------
        Path : [str]
            path to folder that contains the parameter rasters.

        Returns
        -------
        data : [array]
            array contains the average values of the distributed parameters.
        """
        # data type
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check wether the folder has the rasters or not
        assert len(os.listdir(Path)) > 0, Path + " folder you have provided is empty"
        # read data
        data = raster.readRastersFolder(Path)
        # get the No data value from the first raster in the folder
        scr = gdal.Open(f"{Path}/{os.listdir(Path)[0]}")
        _, NoDataValue = raster.getRasterData(scr)
        data[data == NoDataValue] = np.nan

        data = np.nanmean(data, axis=0)
        data = data.mean(0)

        return data

    @staticmethod
    def renameFiles(
        path: str, prefix: str = "", fmt: str = "%Y.%m.%d", freq: str = "daily"
    ):
        """renameFiles.

        renameFiles method takes the path to a folder where you want to put a number
        at the begining of the raster names indicating the order of the raster based on
        its date

        Parameters
        ----------
        path : [str]
            path where the rasters are stored.
        prefix: [str]
            any string you want to add to the raster names, (i.e the dataset name precipitation_ecmwf). Default is "".
        fmt : [String], optional
            the format of the date. The default is '%Y.%m.%d'.
        freq: [str]
            Default is "daily".

        Returns
        -------
        files in the Path are going to have a new name including the order at
        the begining of the name.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("The directory you have entered does not exist")

        files = os.listdir(path)
        # get only the tif files
        files = [i for i in files if i.endswith(".tif")]

        # get the date
        dates_str = [files[i].split("_")[-1][:-4] for i in range(len(files))]
        dates = [dt.datetime.strptime(dates_str[i], fmt) for i in range(len(files))]

        if freq == "daily":
            new_date_str = [
                str(i.year) + "_" + str(i.month) + "_" + str(i.day) for i in dates
            ]
        elif freq == "hourly":
            new_date_str = [
                str(i.year) + "_" + str(i.month) + "_" + str(i.day) + "_" + str(i.hour)
                for i in dates
            ]
        else:
            new_date_str = [
                f"{i.year}-{i.month}-{i.day}-{i.hour}-{i.minute}" for i in dates
            ]

        df = pd.DataFrame()
        df["files"] = files
        df["DateStr"] = new_date_str
        df["dates"] = dates
        df.sort_values("dates", inplace=True)
        df.reset_index(inplace=True)
        df["order"] = [i for i in range(len(files))]

        df["new_names"] = [
            f"{df.loc[i, 'order']}_{prefix}_{df.loc[i, 'DateStr']}.tif"
            for i in range(len(files))
        ]
        # rename the files
        for i in range(len(files)):
            os.rename(
                f"{path}/{df.loc[i, 'files']}", f"{path}/{df.loc[i, 'new_names']}"
            )

    @staticmethod
    def changetext2time(string):
        """changetext2time.

        this functions changes the date from a string to a date time
        format
        """
        time = dt.datetime(
            int(string[:4]),
            int(string[5:7]),
            int(string[8:10]),
            int(string[11:13]),
            int(string[14:16]),
            int(string[17:]),
        )
        return time

    @staticmethod
    def ReadExcelData(path, years, months):
        """ReadExcelData.

            this function reads data listed in excel sheet with years and months are
            listed as columns and days are listed in the first row
            year month 1 2 3 4 5 6 7 8 9 .....................31
            2012  1    5 6 2 6 8 6 9 7 4 3 ...................31
            2012  2    9 8 7 6 3 2 1 5 5 9 ...................31

        Parameters
        ----------
        path:
            [string] path of the excel file
        years:
            [list] list of the years you want to read
        months:
            [list] list of the months you you want to read

        Returns
        -------
         List of the values in the excel file

        examples
        --------
            years=[2009,2010,2011]#,2012,2013]
            months=[1,2,3,4,5,6,7,8,9,10,11,12]
            Q=ReadExcelData(path+"Discharge/Qout.xlsx",years,months)
        """

        Qout = pd.read_excel(path)
        Q = []
        #    years=[2009,2010,2011]#,2012,2013]
        #    months=[1,2,3,4,5,6,7,8,9,10,11,12]
        for year in years:
            for month in months:
                row = Qout[Qout["year"] == year][Qout["month"] == month]
                row = row.drop(["year", "month"], axis=1)
                row = row.values.tolist()[0]
                Q = Q + row

        Q = [Q[i] for i in range(len(Q)) if not np.isnan(Q[i])]

        return Q

    def ListAttributes(self):
        """Print Attributes List."""
        logger.info("\n")
        logger.info(
            f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                logger.info(f"{key} : {repr(self.__dict__[key])}")

        logger.info("\n")
