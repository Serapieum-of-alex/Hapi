"""Rainfall-runoff Inputs."""
from typing import Union
from pathlib import Path
import datetime as dt
import os
import pandas as pd
from geopandas import GeoDataFrame
from pyramids.dataset import Dataset, Datacube

import Hapi

PARAMETERS_LIST = [
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


class Inputs:
    """Rainfall-runoff Inputs class.

        Inputs class contains methods to prepare the inputs for the distributed
        hydrological model

    Methods
    -------
        1- prepareInputs
        2- extractParametersBoundaries
        3- extractParameters
        4- createLumpedInputs
        5- renameFiles
        8- ListAttributes
    """

    def __init__(self, src: str):
        """Rainfall Inputs.

        Parameters
        ----------
        src: [str]
            path to the spatial information source raster to get the spatial information
            (coordinate system, no of rows & columns) A_path should include the name of the raster
            and the extension like "data/dem.tif".
        """
        self.source_dem = src
        pass

    def prepare_inputs(
        self, input_folder: Union[str, Path], output_folder: Union[str, Path]
    ):
        """prepareInputs.

        this function prepares downloaded raster data to have the same alignment and
        nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
        and returns a folder with the output rasters with a name "New_Rasters"

        Parameters
        ----------
        input_folder: [str/Path]
            path of the folder of the rasters you want to adjust their no of rows, columns and resolution (alignment)
            like a source raster.
        output_folder: [str]
            name to create a folder to store resulted rasters.

        Example
        -------
        Ex1:
            >>> dem_path = "01GIS/inputs/4000/acc4000.tif"
            >>> prec_in_path = "02Precipitation/CHIRPS/Daily/"
            >>> In = Inputs(dem_path)
            >>> In.prepare_inputs(prec_in_path, "prec")
        Ex2:
            >>> dem_path="01GIS/inputs/4000/acc4000.tif"
            >>> output_path="00inputs/meteodata/4000/"
            >>> evap_in_path="03Weather_Data/evap/"
            >>> In = Inputs(dem_path)
            >>> Inputs.prepare_inputs(evap_in_path, f"{output_path}/evap")
        """
        if not isinstance(output_folder, str):
            print("output_folder input should be string type")

        mask = Dataset.read_file(self.source_dem)
        cube = Datacube.read_multiple_files(input_folder, with_order=False)
        cube.open_datacube()
        cube.align(mask)
        cube.crop(mask, inplace=True)
        path = [f"{output_folder}/{file.split('/')[-1]}" for file in cube.files]
        cube.to_file(path)

    @staticmethod
    def extract_parameters_boundaries(basin: GeoDataFrame):
        """extractParametersBoundaries.

        extractParametersBoundaries

        Parameters
        ----------
        basin: [GeoDataFrame]
            catchment polygon, make sure that the geodataframe contains one row only, if not merge all the polygons
            in the shapefile first.

        Returns
        -------
        ub: [list]
            list of the upper bound of the parameters.
        lb: [list]
            list of the lower bound of the parameters.

        the parameters are
            ["tt", "sfcf","cfmax","cwh","cfr","fc","beta",
             "lp","k0","k1","k2","uzl","perc", "maxbas"]
        """
        parameters_path = f"{os.path.dirname(Hapi.__file__)}/parameters"

        dataset = Dataset.read_file(f"{parameters_path}/max/{PARAMETERS_LIST[0]}.tif")
        basin = basin.to_crs(crs=dataset.crs)
        # max values
        ub = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(
                f"{parameters_path}/max/{PARAMETERS_LIST[i]}.tif"
            )
            vals = dataset.stats(mask=basin)
            ub.append(vals.loc[vals.index[0], "max"])

        # min values
        lb = list()
        for i in range(len(PARAMETERS_LIST)):
            dataset = Dataset.read_file(
                f"{parameters_path}/min/{PARAMETERS_LIST[i]}.tif"
            )
            vals = dataset.stats(mask=basin)
            lb.append(vals.loc[vals.index[0], "min"])

        par = pd.DataFrame(index=PARAMETERS_LIST)

        par["ub"] = ub
        par["lb"] = lb

        return par

    def extract_parameters(
        self,
        gdf: Union[GeoDataFrame, str],
        scenario: str,
        as_raster: bool = False,
        save_to: str = "",
    ):
        """extract_parameters.

        extractParameters method extracts the parameter raster at the location
        of the source raster, there are 12 set of parameters 10 sets of parameters
        (Beck et al., (2016)) and the max, min and average of all sets


        Beck, H. E., Dijk, A. I. J. M. van, Ad de Roo, Diego G. Miralles,
        T. R. M. & Jaap Schellekens, and L. A. B. (2016) Global-scale
        regionalization of hydrologic model parameters-Supporting materials
        3599–3622. doi:10.1002/2015WR018247.Received

        Parameters
        ----------
        gdf: [GeoDataFrame]
            geodataframe of catchment polygon, make sure that the geodataframe contains
            one row only, if not merge all the polygons in the shapefile first.
        scenario: [str]
            name of the parameter set, there are 12 sets of parameters
            ["1","2","3","4","5","6","7","8","9","10","avg","max","min"]
        as_raster: [bool]
            Default is False.
        save_to: [str]
            path to a directory where you want to save the raster's.

        Returns
        -------
        Parameters : [list]
            list of the upper bound of the parameters.


        the parameters are
            ["tt", rfcf,"sfcf","cfmax","cwh","cfr","fc","beta",'etf'
             "lp","k0","k1","k2","uzl","perc", "maxbas",'K_muskingum',
             'x_muskingum']
        """
        parameters_path = os.path.dirname(Hapi.__file__)
        parameters_path = f"{parameters_path}/parameters/{scenario}"

        if not as_raster:
            dataset = Dataset.read_file(f"{parameters_path}/{PARAMETERS_LIST[0]}.tif")
            gdf = gdf.to_crs(crs=dataset.crs)

            stats = pd.DataFrame(columns=["min", "max", "mean", "std"])
            for i in range(len(PARAMETERS_LIST)):
                dataset = Dataset.read_file(
                    f"{parameters_path}/{PARAMETERS_LIST[i]}.tif"
                )
                vals = dataset.stats(mask=gdf)
                stats.loc[PARAMETERS_LIST[i], :] = vals.loc[
                    :, ["min", "max", "mean", "std"]
                ].values
            return stats
        else:
            self.prepare_inputs(f"{parameters_path}/", save_to)

    @staticmethod
    def create_lumped_inputs(
        path: str,
        regex_string=r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str = None,
        start: str = None,
        end: str = None,
        fmt: str = "%Y-%m-%d",
        extension: str = ".tif",
    ) -> list:
        """create_lumped_inputs.

        create_lumped_inputs method generates lumped parameters from distributed parameters by taking the average

        Parameters
        ----------
        path: [str]
            path to folder that contains the parameter rasters.
        regex_string: [str]
            a regex string that we can use to locate the date in the file names.Default is
            r"d{4}.d{2}.d{2}".
            >>> fname = 'MSWEP_YYYY.MM.DD.tif'
            >>> regex_string = r'd{4}.d{2}.d{2}'
            - or
            >>> fname = 'MSWEP_YYYY_M_D.tif'
            >>> regex_string = r'd{4}_d{1}_d{1}'
            - if there is a number at the beginning of the name
            >>> fname = '1_MSWEP_YYYY_M_D.tif'
            >>> regex_string = r'd+'
        date: [bool]
            True if the number in the file name is a date. Default is True.
        file_name_data_fmt : [str]
            if the files' names have a date, and you want to read them ordered .Default is None
            >>> "MSWEP_YYYY.MM.DD.tif"
            >>> file_name_data_fmt = "%Y.%m.%d"
        start: [str]
            start date if you want to read the input raster for a specific period only and not all rasters,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        fmt: [str]
            format of the given date in the start/end parameter.
        extension: [str]
            the extension of the files you want to read from the given path. Default is ".tif".

        Returns
        -------
        List:
            list containing the average values of the distributed parameters.
        """
        cube = Datacube.read_multiple_files(
            path,
            with_order=True,
            regex_string=regex_string,
            date=date,
            start=start,
            end=end,
            fmt=fmt,
            file_name_data_fmt=file_name_data_fmt,
            extension=extension,
        )
        cube.open_datacube()
        avg = []
        for i in range(cube.time_length):
            dataset = cube.iloc(i)
            stats = dataset.stats()
            avg.append(stats.loc[stats.index[0], "mean"])

        return avg

    @staticmethod
    def rename_files(
        path: str, prefix: str = "", fmt: str = "%Y.%m.%d", freq: str = "daily"
    ):
        """renameFiles.

        renameFiles method takes the path to a folder where you want to put a number
        at the beginning of the raster names indicating the order of the raster based on
        its date

        Parameters
        ----------
        path : [str]
            path where the rasters are stored.
        prefix: [str]
            any string you want to add to the raster names, (i.e., the dataset name precipitation_ecmwf). Default is "".
        fmt: [String], optional
            the format of the date. The default is '%Y.%m.%d'.
        freq: [str]
            Default is "daily".

        Returns
        -------
        files in the Path are going to have a new name including the order at the beginning of the name.
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
