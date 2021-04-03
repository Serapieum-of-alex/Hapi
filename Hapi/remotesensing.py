# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 06:58:20 2021

@author: mofarrag
"""
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime as dt
import calendar
from ecmwfapi import ECMWFDataServer

import Hapi.data_conversions as DC
import Hapi.weirdFn as weirdFn

class RemoteSensing():
    
    def __init__():
        pass
    
    def main(Dir, Vars, Startdate, Enddate, latlim, lonlim, cores=False,
             SumMean=1, Min=0, Max=0, Waitbar = 1):
        """
        This function downloads ECMWF daily data for a given variable, time
        interval, and spatial extent.
    
        Keyword arguments:
        Dir -- 'C:/file/to/path/'
        Var -- Variable code: VariablesInfo('day').descriptions.keys()
        Startdate -- 'yyyy-mm-dd'
        Enddate -- 'yyyy-mm-dd'
        latlim -- [ymin, ymax]
        lonlim -- [xmin, xmax]
        SumMean -- 0 or 1. Indicates if the output values are the daily mean for
                   instantaneous values or sum for fluxes
        Min -- 0 or 1. Indicates if the output values are the daily minimum
        Max -- 0 or 1. Indicates if the output values are the daily maximum
        Waitbar -- 1 (Default) will create a waitbar
        """
        for Var in Vars:
    		# Download data
            print('\nDownload ECMWF %s data for period %s till %s' %(Var, Startdate, Enddate))
            
            RemoteSensing.DownloadData(Dir, Var, Startdate, Enddate, latlim, lonlim, Waitbar, cores,
    					 TimeCase='daily', CaseParameters=[SumMean, Min, Max])
    
        del_ecmwf_dataset = os.path.join(Dir,'data_interim.nc')
        os.remove(del_ecmwf_dataset)
    
    
    def DownloadData(Dir, Var, Startdate, Enddate, latlim, lonlim, Waitbar, cores,
                     TimeCase, CaseParameters):
        """
        This function downloads ECMWF six-hourly, daily or monthly data
    
        Keyword arguments:
    
        """
    
        # correct latitude and longitude limits
        latlim_corr_one = np.floor(latlim[0]/0.125) * 0.125
        latlim_corr_two = np.ceil(latlim[1]/0.125) * 0.125
        latlim_corr = [latlim_corr_one, latlim_corr_two]
    
        # correct latitude and longitude limits
        lonlim_corr_one = np.floor(lonlim[0]/0.125) * 0.125
        lonlim_corr_two = np.ceil(lonlim[1]/0.125) * 0.125
        lonlim_corr = [lonlim_corr_one, lonlim_corr_two]
    
        # Load factors / unit / type of variables / accounts
        VarInfo = VariablesInfo(TimeCase)
        Varname_dir = VarInfo.file_name[Var]
    
        # Create Out directory
        out_dir = os.path.join(Dir, "Weather_Data", "Model", "ECMWF", TimeCase, Varname_dir, "mean")
        if not os.path.exists(out_dir):
              os.makedirs(out_dir)
    
        DownloadType = VarInfo.DownloadType[Var]
    
        # Set required data for the three hourly option
        if TimeCase == 'six_hourly':
            string1 = 'oper'
    
        # Set required data for the daily option
        elif TimeCase == 'daily':
            Dates = pd.date_range(Startdate,  Enddate, freq='D')
        elif TimeCase == 'monthly':
            Dates = pd.date_range(Startdate,  Enddate, freq='MS')
    
        if DownloadType == 1:
            string1 = 'oper'
            string4 = "0"
            string6 = "00:00:00/06:00:00/12:00:00/18:00:00"
            string2 = 'sfc'
            string8 = 'an'
    
        if DownloadType == 2:
            string1 = 'oper'
            string4 = "12"
            string6 = "00:00:00/12:00:00"
            string2 = 'sfc'
            string8 = 'fc'
    
        if DownloadType == 3:
            string1 = 'oper'
            string4 = "0"
            string6 = "00:00:00/06:00:00/12:00:00/18:00:00"
            string2 = 'pl'
            string8 = 'an'
    
        string7 = '%s/to/%s'  %(Startdate, Enddate)
    
        parameter_number = VarInfo.number_para[Var]
        string3 = '%03d.128' %(parameter_number)
        string5 = '0.125/0.125'
        string9 = 'ei'
        string10 = '%s/%s/%s/%s' %(latlim_corr[1], lonlim_corr[0], latlim_corr[0], lonlim_corr[1])   #N, W, S, E
    
    
        # Download data by using the ECMWF API
        
        print('Use API ECMWF to collect the data, please wait')
        API(Dir, DownloadType, string1, string2, string3, string4, string5, string6, string7, string8, string9, string10)
    
        # Open the downloaded data
        NC_filename = os.path.join(Dir,'data_interim.nc')
        fh = Dataset(NC_filename, mode='r')
    
        # Get the NC variable parameter
        parameter_var = VarInfo.var_name[Var]
        Var_unit = VarInfo.units[Var]
        factors_add = VarInfo.factors_add[Var]
        factors_mul = VarInfo.factors_mul[Var]
    
        # Open the NC data
        Data = fh.variables[parameter_var][:]
        Data_time = fh.variables['time'][:]
        lons = fh.variables['longitude'][:]
        lats = fh.variables['latitude'][:]
    
        # Define the georeference information
        Geo_four = np.nanmax(lats)
        Geo_one = np.nanmin(lons)
        Geo_out = tuple([Geo_one, 0.125, 0.0, Geo_four, 0.0, -0.125])
    
        # Create Waitbar
        if Waitbar == 1:
            total_amount = len(Dates)
            amount = 0
            weirdFn.printWaitBar(amount, total_amount, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
        for date in Dates:
    
            # Define the year, month and day
            year =  date.year
            month =  date.month
            day =  date.day
    
            # Hours since 1900-01-01
            start = dt.datetime(year=1900, month=1, day=1)
            end = dt.datetime(year, month, day)
            diff = end - start
            hours_from_start_begin = diff.total_seconds()/60/60
    
            Date_good = np.zeros(len(Data_time))
            if TimeCase == 'daily':
                 days_later = 1
            if TimeCase == 'monthly':
                 days_later = calendar.monthrange(year,month)[1]
    
            Date_good[np.logical_and(Data_time>=hours_from_start_begin, Data_time<(hours_from_start_begin + 24 * days_later))] = 1
    
            Data_one = np.zeros([int(np.sum(Date_good)),int(np.size(Data,1)),int(np.size(Data,2))])
            Data_one = Data[np.int_(Date_good) == 1, :, :]
    
            # Calculate the average temperature in celcius degrees
            Data_end = factors_mul * np.nanmean(Data_one,0) + factors_add
    
            if VarInfo.types[Var] == 'flux':
                Data_end = Data_end * days_later
    
            VarOutputname = VarInfo.file_name[Var]
    
            # Define the out name
            name_out = os.path.join(out_dir, "%s_ECMWF_ERA-Interim_%s_%s_%d.%02d.%02d.tif" %(VarOutputname, Var_unit, TimeCase, year,month,day))
    
            # Create Tiff files
            DC.Save_as_tiff(name_out, Data_end, Geo_out, "WGS84")
    
            if Waitbar == 1:
                amount += 1
                weirdFn.printWaitBar(amount, total_amount, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    
        fh.close()
    
        return()
    
    def API(output_folder, DownloadType, string1, string2, string3, string4, string5, string6, string7, string8, string9, string10):
    
    
    	server = ECMWFDataServer()

	    if DownloadType == 1 or DownloadType == 2:
	        server.retrieve({
	            'stream'    : "%s" %string1,
	            'levtype'   : "%s" %string2,
	            'param'     : "%s" %string3,
	            'dataset'   : "interim",
	            'step'      : "%s" %string4,
	            'grid'      : "%s" %string5,
	            'time'      : "%s" %string6,
	            'date'      : "%s" %string7,
	            'type'      : "%s" %string8,     # http://apps.ecmwf.int/codes/grib/format/mars/type/
	            'class'     : "%s" %string9,     # http://apps.ecmwf.int/codes/grib/format/mars/class/
	            'area'      : "%s" %string10,   							
	            'format'    : "netcdf",
	            'target'    : "data_interim.nc"
	            })

	    if DownloadType == 3:
	        server.retrieve({
	            'levelist'   : "1000",
	            'stream'    : "%s" %string1,
	            'levtype'   : "%s" %string2,
	            'param'     : "%s" %string3,
	            'dataset'   : "interim",
	            'step'      : "%s" %string4,
	            'grid'      : "%s" %string5,
	            'time'      : "%s" %string6,
	            'date'      : "%s" %string7,
	            'type'      : "%s" %string8,     # http://apps.ecmwf.int/codes/grib/format/mars/type/
	            'class'     : "%s" %string9,     # http://apps.ecmwf.int/codes/grib/format/mars/class/
	            'area'      : "%s" %string10,   							
	            'format'    : "netcdf",
	            'target'    : "data_interim.nc"
	            })
		
		
	    return()


class VariablesInfo:
    """
    This class contains the information about the ECMWF variables
    http://rda.ucar.edu/cgi-bin/transform?xml=/metadata/ParameterTables/WMO_GRIB1.98-0.128.xml&view=gribdoc
    """
    number_para = {'T'  : 130,
                   '2T'  : 167,
                   'SRO' : 8,
                   'SSRO' :9,
                   'WIND' : 10,
                   '10SI' : 207,
                   'SP' : 134,
                   'Q' : 133,
                   'SSR': 176,
                   'R': 157,
                   'E':  182,
                   'SUND': 189,
                   'RO' : 205,
                   'TP' : 228,
                   '10U' : 165,
                   '10V' : 166,
                   '2D' : 168,
                   'SR' : 173,
                   'AL' : 174,
                   'HCC': 188}

    var_name = {'T'  : 't',
                '2T'  : 't2m',
                'SRO' : 'sro',
                'SSRO' :'ssro',
                'WIND' : 'wind',
                '10SI' : '10si',
                'SP' :'sp',
                'Q' : 'q',
                'SSR': 'ssr',
                'R': 'r',
                'E':  'e',
                'SUND': 'sund',
                'RO' : 'ro',
                'TP' : 'tp',
                '10U' : 'u10',
                '10V' : 'v10',
                '2D' : 'd2m',
                'SR' : 'sr',
                'AL' : 'al',
                'HCC': 'hcc'}

	# ECMWF data
    descriptions = {'T'  : 'Temperature [K]',
                    '2T': '2 meter Temperature [K]',
                    'SRO' : 'Surface Runoff [m]',
                    'SSRO' :'Sub-surface Runoff [m]',
                    'WIND' : 'Wind speed [m s-1]',
                    '10SI' : '10 metre windspeed [m s-1]',
                    'SP' :'Surface Pressure [pa]',
                    'Q' : 'Specific humidity [kg kg-1]',
                    'SSR': 'Surface solar radiation [W m-2 s]',
                    'R': 'Relative humidity [%]',
                    'E':  'Evaporation [m of water]',
                    'SUND': 'Sunshine duration [s]',
                    'RO' : 'Runoff [m]',
                    'TP' : 'Total Precipitation [m]',
                    '10U' : '10 metre U wind component [m s-1]',
                    '10V' : '10 metre V wind component [m s-1]',
                    '2D' : '2 metre dewpoint temperature [K]',
                    'SR' : 'Surface roughness [m]',
                    'AL' : 'Albedo []',
                    'HCC': 'High cloud cover []'}

    # Factor add to get output
    factors_add = {'T': -273.15,
                   '2T': -273.15,
                   'SRO' : 0,
                   'SSRO' :0,
                   'WIND' : 0,
                   '10SI' : 0,
                   'SP' : 0,
                   'Q' : 0,
                   'SSR' : 0,
                   'R': 0,
                   'E': 0,
                   'SUND': 0,
                   'RO' : 0,
                   'TP' : 0,
                   '10U' : 0,
                   '10V' : 0,
                   '2D' : -273.15,
                   'SR' : 0,
                   'AL' : 0,
                   'HCC': 0}

    # Factor multiply to get output
    factors_mul = {'T': 1,
                   '2T': 1,
                   'SRO' : 1000,
                   'SSRO' :1000,
                   'WIND' : 1,
                   '10SI' : 1,
                   'SP' : 0.001,
                   'Q' : 1,
                   'SSR' : 1,
                   'R': 1,
                   'E': 1000,
                   'SUND': 1,
                   'RO' : 1000,
                   'TP' : 1000,
                   '10U' : 1,
                   '10V' : 1,
                   '2D' : 1,
                   'SR' : 1,
                   'AL' : 1,
                   'HCC': 1}

    types = {'T': 'state',
             '2T': 'state',
             'SRO' : 'flux',
             'SSRO' :'flux',
             'WIND' : 'state',
             '10SI' : 'state',
             'SP' : 'state',
             'Q' : 'state',
             'SSR' : 'state',
             'R': 'state',
             'E': 'flux',
             'SUND':'flux',
             'RO' : 'flux',
             'TP' : 'flux',
             '10U' : 'state',
             '10V' : 'state',
             '2D' : 'state',
             'SR' : 'state',
             'AL' : 'state',
             'HCC': 'state'}

    file_name = {'T': 'Tair2m',
                 '2T': 'Tair',
                 'SRO' : 'Surf_Runoff',
                 'SSRO' :'Subsurf_Runoff',
                 'WIND' : 'Wind',
                 '10SI' : 'Wind10m',
                 'SP' : 'Psurf',
                 'Q' : 'Qair',
                 'SSR' : 'SWnet',
                 'R': 'RelQair',
                 'E': 'Evaporation',
                 'SUND':'SunDur',
                 'RO' : 'Runoff',
                 'TP' : 'P',
                 '10U' : 'Wind_U',
                 '10V' : 'Wind_V',
                 '2D' : 'Dewpoint2m',
                 'SR' : 'SurfRoughness',
                 'AL' : 'Albedo',
                 'HCC': 'HighCloudCover'
                 }

    DownloadType = {'T': 3,
                 '2T': 1,
                 'SRO' : 0,
                 'SSRO' : 0,
                 'WIND' : 0,
                 '10SI' : 0,
                 'SP' : 1,
                 'Q' : 3,
                 'SSR' : 2,
                 'R': 3,
                 'E': 2,
                 'SUND':2,
                 'RO' : 2,
                 'TP' : 2,
                 '10U' : 1,
                 '10V' : 1,
                 '2D' : 1,
                 'SR' : 1,
                 'AL' : 1,
                 'HCC': 1
                 }

    def __init__(self, step):

        # output units after applying factor
        if step == 'six_hourly':
            self.units = {'T': 'C',
                          '2T': 'C',
                          'SRO' : 'mm',
                          'SSRO' :'mm',
                          'WIND' : 'm_s-1',
                          '10SI' : 'm_s-1',
                          'SP' : 'kpa',
                          'Q' : 'kg_kg-1',
                          'SSR' : 'W_m-2_s',
                          'R': 'percentage',
                          'E': 'mm',
                          'SUND':'s',
                          'RO' : 'mm',
                          'TP' : 'mm',
                          '10U' : 'm_s-1',
                          '10V' : 'm_s-1',
                          '2D' : 'C',
                          'SR' : 'm',
                          'AL' : '-',
                          'HCC': '-'
                 }

        elif step == 'daily':
            self.units = {'T': 'C',
                          '2T': 'C',
                          'SRO' : 'mm',
                          'SSRO' :'mm',
                          'WIND' : 'm_s-1',
                          '10SI' : 'm_s-1',
                          'SP' : 'kpa',
                          'Q' :'kg_kg-1',
                          'SSR' : 'W_m-2_s',
                          'R': 'percentage',
                          'E': 'mm',
                          'SUND':'s',
                          'RO' : 'mm',
                          'TP' : 'mm',
                          '10U' : 'm_s-1',
                          '10V' : 'm_s-1',
                          '2D' : 'C',
                          'SR' : 'm',
                          'AL' : '-',
                          'HCC': '-'}

        elif step == 'monthly':
            self.units = {'T': 'C',
                          '2T': 'C',
                          'SRO' : 'mm',
                          'SSRO' :'mm',
                          'WIND' : 'm_s-1',
                          '10SI' : 'm_s-1',
                          'SP' : 'kpa',
                          'Q' :'kg_kg-1',
                          'SSR' : 'W_m-2_s',
                          'R': 'percentage',
                          'E': 'mm',
                          'SUND':'s',
                          'RO' : 'mm',
                          'TP' : 'mm',
                          '10U' : 'm_s-1',
                          '10V' : 'm_s-1',
                          '2D' : 'C',
                          'SR' : 'm',
                          'AL' : '-',
                          'HCC': '-'}

        else:
            raise KeyError("The input time step is not supported")
