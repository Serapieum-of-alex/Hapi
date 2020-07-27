#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:27:24 2020

@author: juanmanuel
"""

from Hapi.raster import RastersSeriesFromPointsSHPtoXLSX

start_date = "01-01-2000"
end_date = "31-12-2015"

shp_filename = '/Users/juanmanuel/Documents/Juan Manuel/Universidad/TESIS/Datos/GIS/CALIBRATION_POINTS/CALIBRATION_POINTS.shp'
SHPField_name = 'id'
rasters_path = '/Users/juanmanuel/Documents/Juan Manuel/Universidad/TESIS/Datos/meteodata/calib/flow/'
file_first_str = "daily-flow_"
file_second_str = ".tif"
date_format = "%Y%m%d"
output_file = '/Users/juanmanuel/Documents/Juan Manuel/Universidad/TESIS/Datos/DBs/Qobs.xlsx'


RastersSeriesFromPointsSHPtoXLSX(start_date, 
                                    end_date, 
                                    shp_filename, 
                                    SHPField_name, 
                                    rasters_path, 
                                    file_first_str, 
                                    file_second_str, 
                                    date_format, 
                                    output_file)