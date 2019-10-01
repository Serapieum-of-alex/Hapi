"""
main page
"""
import os
os.chdir("F:/01Algorithms/HAPI/Web_application")

import sys
sys.path.append("HBV_distributed/function")
import gdal
# to add the gdal data environment variable
gdal.SetConfigOption("GDAL_DATA","E:/Anaconda2/envs/Env_64_2.7/Library/share/gdal")
# to add the proj data environment variable
gdal.SetConfigOption("PROJ_data","E:/Anaconda2/envs/Env_64_2.7/Library/share/epsg_csv")
#%% Library
import numpy as np
import pandas as pd
import datetime as dt

from math import pi
import StringIO
import base64
import geopandas as gpd
from shapely.geometry import Polygon
from fiona.crs import from_epsg
import osr
from collections import OrderedDict
#import pysal as ps
#from datetime import datetime,date
#import time

# bokeh
from bokeh.layouts import layout, widgetbox, row, column,  gridplot #,Widget
from bokeh.models.widgets import (Panel, Button, TextInput, Div, Tabs ,Slider,Select ,
                                  RadioButtonGroup, #DataTable, DateFormatter, TableColumn, 
#                                 DateRangeSlider, DateFormatter, DataTable, TableColumn #,DatePicker, NumberFormatter 
                                  )
from bokeh.models import (ColumnDataSource, CustomJS, GMapPlot,GMapOptions, 
                          LinearAxis, Range1d, HoverTool, PanTool, WheelZoomTool,
                          BoxSelectTool, ColorBar,LogColorMapper,ResetTool,BoxZoomTool, #SaveTool,
                          CrosshairTool,
                         # NumeralTickFormatter, #PrintfTickFormatter, #BoxSelectionOverlay 
                          Circle, Square, Title, Legend) #, Slider,GeoJSONDataSource, PreviewSaveTool,

from bokeh.plotting import figure#, gmap
from bokeh.io import curdoc , show
#from bokeh.palettes import YlOrRd6 as palette
#from bokeh.palettes import RdYlGn10 as palette
from bokeh.models.glyphs import Patches #, Line, Circle

#from bokeh.core import templates
#from bokeh.resources import CDN
#from bokeh.embed import components, autoload_static, autoload_server
#from bokeh.plotting import save

# functions
#import DHBV_functions
import GISpy as GIS
import WeirdFn
import DistParameters 
import Wrapper
import PerformanceCriteria
import plotting_functions as plf
import java_functions
import Inputs as pf
import StatisticalTools as st
#from inputs import Inputs
#%% Run the model
# Read the input data 
data_file= 'HBV_distributed/static/input_data/' # Name of the output file

s=dt.datetime(2012,6,14,19,0,0)
e=dt.datetime(2013,12,23,0,0,0)

index=pd.date_range(s,e,freq="1H")
lake_data=pd.DataFrame(index=index)

# Read data from the output file
lake_data['et']=np.loadtxt(data_file+"lake/" + "et.txt")
lake_data['t']=np.loadtxt(data_file+"lake/" + "temp.txt")
lake_data['tm']=np.loadtxt(data_file+"lake/" + "avgtemp.txt")
lake_data['plake']=np.loadtxt(data_file+"lake/" + "plake.txt")
lake_data['Q']=np.loadtxt(data_file+"lake/" + "Q.txt")


lake_data_A=lake_data.as_matrix()
curve=np.load(data_file+"curve.npy")
jiboa_initial=np.loadtxt(data_file+"Initia-jiboa.txt",usecols=0)
lake_initial=np.loadtxt(data_file+"Initia-lake.txt",usecols=0)

lakecell=[2,1] # 4km
#lakecell=[4,2] # 2km
#lakecell=[10,4] # 1km
#lakecell=[19,10] # 500m

sp_prec_c=np.load(data_file +'sp_prec_c.npy')
sp_et_c=np.load(data_file +'sp_et_c.npy')
sp_temp_c=np.load(data_file +'sp_temp_c.npy')

sp_quz_4km=np.load(data_file + "q_uz_c_4km.npy")

flow_acc_table=WeirdFn.load_obj(data_file +"flow_acc_table")
flow_acc=np.load(data_file +'flow_acc.npy')

DEM = gdal.Open(data_file+"/DEM/"+"dem_4km.tif")
elev, no_val=GIS.GetRasterData(DEM)

elev[elev==no_val]=np.nan
no_cells=np.size(elev[:,:])-np.count_nonzero(np.isnan(elev[:,:]))

# Create vector with time stamps
#time_index = pd.date_range('1994 12 07 20:00', periods=len(data), freq='H')
#data.set_index(time_index, inplace=True)

# Intial Parameters 
pars =np.loadtxt(data_file +"parameters.txt")
klb=pars [-2]
kub=pars [-1]
pars =pars [:-2]

jiboa_par,lake_par=DistParameters.par2d_lumpedK1_lake(pars,DEM,12,13,kub,klb)

#pars = [0.5, 0.2, 0.01, 0.1, 10.0, 20.0, 1, 1]
extra_pars = [1, 227.31,133.98,70.64] # [time factor,catchment area, lakecatchment area, lake area]

# Define the precipitation data to give to the model

#prec =sp_prec_c[:,lakecell[0],lakecell[1]]
prec =lake_data['plake']
#evap =sp_et_c[:,lakecell[0],lakecell[1]]
evap =lake_data['et']
q_rec =lake_data['Q'].tolist()
#snow = np.array(data['Snowfall'])

# Setup model (function)
q_sim = []  
#set up data source
# all input data
ds_rec = ColumnDataSource(dict(q_rec = q_rec, ds_time = index, evap = evap,
                           prec = prec))
# calculated discharge 
q_sim=np.loadtxt(data_file +"Q4km.txt")[0:len(prec)]
ds_sim = ColumnDataSource(dict( q_sim = q_sim , ds_time = index))

# plotting in run model tab

#widget dimensions for plotting
width =  620 # width of the input data graph in sugawara model tab
width2 = 640 # width of the sim vs rec hydrograph in sugawara model tab
height = 430

maxq_rec = np.max(np.array(q_rec)) # maximum discharge from recorded values
maxp = np.max(prec) # maximum precipitation from data

# Precipitation and Recorded Discharge plot 
# setup plot
plot_sim = figure(width=width, 
                     height=height,
                     title="Precipitation and Recorded Discharge",
                     y_range = (0, 1.75*maxq_rec),
                     x_axis_type = "datetime",
                     toolbar_location = "above",)

plot_sim.extra_y_ranges = {"eff_rain": Range1d(start=3.0*maxp, end=0)}

# plot precip
plot_sim.line(x = 'ds_time', 
                 y = 'q_rec', 
                 source = ds_rec,
                 color="navy",
                 legend='recorded discharge')

# plot q recorded
plot_sim.line(x = 'ds_time', 
                 y = 'prec', 
                 source = ds_rec,
                 color="grey",
                 y_range_name="eff_rain",
                 legend='precipitation')

plot_sim.yaxis.axis_label = "Discharge [m3/s]"
plot_sim.xaxis.axis_label = "Dates"
plot_sim.xaxis.major_label_orientation = pi/4

plot_sim.add_layout(LinearAxis(y_range_name="eff_rain" , 
                                  axis_label = "Rainfall [mm]" ), 'right')
#______________________________________________________________________________
#______________________________________________________________________________
# rec vs simulated hydrograph
plot_qsim = figure(width=width2, 
                   height=height,
                   title="Recorded vs Simulated Discharge",
                   toolbar_location = "above",
                   x_axis_type = "datetime")

plot_qsim.line(x = 'ds_time', 
               y = 'q_sim', 
               source = ds_sim, 
               color="firebrick",
               legend='simulated discharge')

plot_qsim.line(x = 'ds_time', 
               y = 'q_rec', 
               source = ds_rec, 
               color="navy",
               legend='recorded discharge')

plot_qsim.yaxis.axis_label = "Discharge [m3/s]"
plot_qsim.xaxis.axis_label = "Dates"
plot_qsim.xaxis.major_label_orientation = pi/4
#______________________________________________________________________________
#______________________________________________________________________________
# plot ET
plot_evap = figure(  width=width, 
                     height=height,
                     title="Evapotranspiration",
                     x_axis_type = "datetime",
                     toolbar_location = "above",)

plot_evap.line(x = 'ds_time', 
                 y = 'evap', 
                 source = ds_rec,
                 color="firebrick",
                 legend='Actual ET')

plot_evap.yaxis.axis_label = "ET [mm/t]"
plot_evap.xaxis.axis_label = "Dates"
plot_evap.xaxis.major_label_orientation = pi/4

# make the widgets
w_dt = TextInput(value = '1.0', title = 'Number of hours in the time step [s]')
w_area = TextInput(value = '147.0', title = 'Catchment area [km2]')

# buttons
run_model_button = Button(label = 'Run model', button_type = 'success' , width = 150)
calibrate_button = Button(label = 'Calibrate model', button_type = 'warning', width = 150)

# define the update
### upload data

# precipitation
#sp_quz_2km=np.load(data_file + "q_uz_c_2km.npy")

dates_div=Div(text="<h3>please enter the start and end date of the input data<h3>", width=500, height=10) #,css_classes=['tem    plates\\styles.css'
# default values of the timeseries
s=dt.datetime(2012,6,14,19,0,0)
e=dt.datetime(2013,12,23,0,0,0)

dates=pd.date_range(s,e,freq="1H").tolist()
dates=[str(i) for i in dates]
date_ds=ColumnDataSource(data=dict(date=dates))

Time_step=Select(title="Time Step", value="Hourly", options=["Hourly", "Daily", "weekly"])
startdate=TextInput(title="Start date:",value=str(s), width=100, height=20)
endtdate=TextInput(title="End date:",value=str(e), width=100, height=20)
empty_div1=Div(text=" ", width=100, height=20)

del dates,s,e

# time step button 
def changetimestep(attr,old,new):
    s=dt.datetime(int(startdate.value[:4]),int(startdate.value[5:7]),int(startdate.value[8:10]),
                  int(startdate.value[11:13]),int(startdate.value[14:16]),int(startdate.value[17:]))
    e=dt.datetime(int(endtdate.value[:4]),int(endtdate.value[5:7]),int(endtdate.value[8:10]),
                  int(endtdate.value[11:13]),int(endtdate.value[14:16]),int(endtdate.value[17:]))
    
    if Time_step.value== "Hourly":
        freq="1H"
    elif Time_step.value== "Daily":
        freq="1D"
    elif Time_step.value== "weekly":
        freq="1W"
    dates=pd.date_range(s,e,freq=freq).tolist()
    dates=[str(i) for i in dates]
    # update the date_ds to the new time series dates
    date_ds.data['date']=dates

Time_step.on_change("value",changetimestep)

data_length=len(date_ds.data['date']) #np.shape(prec_in_file_source.data['file_contents'][0])[2]

# file name 
check_text = TextInput(value = ' ', title = 'check area')
TOOLS="pan, box_zoom, reset, save, crosshair"
# TODO button to upload the DEM
### DEM
#DEM_filename= Div(text = 'Temperature file',width=120, height=20)
#DEM_message=Div(text=" upload your data ", width=500, height=10)
#
## create a data source with the name and content stored in a dict
#DEM_in_file_source = ColumnDataSource({'file_contents':[], 'file_name':["anyname"]})
#
##DEM_in_file_source.data['file_contents'][0]=sp_temp_c
#
#def DEM_file_callback(attr,old,new):
#    raw_contents = DEM_in_file_source.data['file_contents']
#    print("raw_contents :"+str(type(raw_contents)))
#    print(raw_contents)
##    prefix, b64_contents = raw_contents.split(",", 1)
##    print("prefix: "+str(type(prefix)))
##    print("b64_contents: "+str(type(b64_contents)))
##    # decode the values from base64 to decimals
##    file_contents = base64.b64decode(b64_contents)
##    print("file_contents: "+str(type(file_contents)))
##    print(file_contents)
#    # now file_contents is exactly the shape of the data in the file to read it use stringIO
#    # file_contents is a  string
##    file_io = StringIO.StringIO(file_contents)
##    print(file_io)
##    data_numpyarray=np.load(file_io)
#    # validation of the length of time series
#    
##    if len(date_ds.data['date']) != np.shape(data_numpyarray)[2]:
##        DEM_message.text=" Temperature data lenght does not match with star, end date and time step"
##    else:
##        temp_message.text=" date matches with length of precipitation input data"
##    DEM= gdal.Open(file_contents)#.ReadAsArray().shape
##    print(str(type(DEM)))
##    print(shape_base_dem.ReadAsArray().shape)
##    DEM_in_file_source.data['file_contents'][0]=file_contents
##    print(file_contents)
##    # get just the name of the file without the extension 
##    DEM_in_file_source.data['file_name'][0]=DEM_in_file_source.data['file_name'][0].split(".")[0]
##    # write the file name
##    DEM_filename.text=DEM_in_file_source.data['file_name'][0]
#
#DEM_in_file_source.on_change('data', DEM_file_callback)
#
#DEM_upload_button = Button(label="Upload-DEM", button_type="success", width = 230,height=20)
#DEM_upload_button.callback = CustomJS(args=dict(file_source=DEM_in_file_source), code =java_functions.raster_upload())

"""temp figure """
# flip the array as the column datasource already flip it 
#DEM_in_plot_source = ColumnDataSource(data=dict(image=[np.flipud(DEM_in_file_source.data['file_contents'][0][:, :, 0])]))
## color bar
#color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(DEM_in_file_source.data['file_contents'][0]),
#                            high=np.nanmax(DEM_in_file_source.data['file_contents'][0]), nan_color="white")
#color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
#                      border_line_color=None, location=(0,0)) #label_standoff=12
## figure
#DEM_input_plot= figure( x_range=(0, np.shape(DEM_in_file_source.data['file_contents'][0])[1]), y_range=(0, np.shape(DEM_in_file_source.data['file_contents'][0])[0]),
#            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
#DEM_input_plot.image(image='image', x=0, y=0, dw=np.shape(DEM_in_file_source.data['file_contents'][0])[1], dh=np.shape(DEM_in_file_source.data['file_contents'][0])[0],
#         source=DEM_in_plot_source ,color_mapper=color_mapper)
#DEM_input_plot.add_layout(color_bar, 'right')
### Temperature
temp_filename= Div(text = 'Temperature file',width=120, height=20)
temp_message=Div(text=" upload your data ", width=500, height=10)

# create a data source with the name and content stored in a dict
temp_in_file_source = ColumnDataSource({'file_contents':[0], 'file_name':["anyname"]})
temp_in_file_source.data['file_contents'][0]=sp_temp_c

def temp_file_callback(attr,old,new):
    raw_contents = temp_in_file_source.data['file_contents'][0]
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
    # now file_contents is exactly the shape of the data in the file to read it use stringIO
    # file_contents is a  string
    file_io = StringIO.StringIO(file_contents)
    data_numpyarray=np.load(file_io)
    # validation of the length of time series
    
    if len(date_ds.data['date']) != np.shape(data_numpyarray)[2]:
        temp_message.text=" Temperature data lenght does not match with star, end date and time step"
    else:
        temp_message.text=" date matches with length of precipitation input data"

    temp_in_file_source.data['file_contents'][0]=data_numpyarray
    # get just the name of the file without the extension 
    temp_in_file_source.data['file_name'][0]=temp_in_file_source.data['file_name'][0].split(".")[0]
    # write the file name
    temp_filename.text=temp_in_file_source.data['file_name'][0]

temp_in_file_source.on_change('data', temp_file_callback)

temp_upload_button = Button(label="Upload-Distributed Temperature", button_type="success", width = 230,height=20)
temp_upload_button.callback = CustomJS(args=dict(file_source=temp_in_file_source), code =java_functions.javaupload())

"""temp figure """
# flip the array as the column datasource already flip it 
temp_in_plot_source = ColumnDataSource(data=dict(image=[np.flipud(temp_in_file_source.data['file_contents'][0][:, :, 0])]))
# color bar
color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(temp_in_file_source.data['file_contents'][0]),
                            high=np.nanmax(temp_in_file_source.data['file_contents'][0]), nan_color="white")
color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                      border_line_color=None, location=(0,0)) #label_standoff=12
# figure
temp_input_plot= figure( x_range=(0, np.shape(temp_in_file_source.data['file_contents'][0])[1]), y_range=(0, np.shape(temp_in_file_source.data['file_contents'][0])[0]),
            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
temp_input_plot.image(image='image', x=0, y=0, dw=np.shape(temp_in_file_source.data['file_contents'][0])[1], dh=np.shape(temp_in_file_source.data['file_contents'][0])[0],
         source=temp_in_plot_source ,color_mapper=color_mapper)
temp_input_plot.add_layout(color_bar, 'right')

# TODO change the tag of the slider to day, hour, month and year 
temp_slider= Slider(title="Evapotranspiration index ", start=0, end=(data_length-1), value=0, step=1, )#title="Frame"
temp_date = TextInput(value = str(date_ds.data['date'][temp_slider.value]), title = 'date:',
                              width=60)
#
def update_temp(attr, old, new):
    # update the plot datasource
    temp_in_plot_source.data = dict(image=[np.flipud(temp_in_file_source.data['file_contents'][0][:, :, temp_slider.value])])
    # update the date text 
    temp_date.value=str(date_ds.data['date'][temp_slider.value])

temp_slider.on_change('value', update_temp)


### evapotranspiration
evap_filename= Div(text = 'Evapotranspiration file',width=120, height=20)
evap_message=Div(text=" upload your data ", width=500, height=10)

# create a data source with the name and content stored in a dict
evap_in_file_source = ColumnDataSource({'file_contents':[0], 'file_name':["anyname"]})
evap_in_file_source.data['file_contents'][0]=sp_et_c

def evap_file_callback(attr,old,new):
    raw_contents = evap_in_file_source.data['file_contents'][0]
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
    # now file_contents is exactly the shape of the data in the file to read it use stringIO
    # file_contents is a  string
    file_io = StringIO.StringIO(file_contents)
    data_numpyarray=np.load(file_io)
    # validation of the length of time series
    
    if len(date_ds.data['date']) != np.shape(data_numpyarray)[2]:
        evap_message.text=" evapotranspiration data lenght does not match with star, end date and time step"
    else:
        evap_message.text=" date matches with length of precipitation input data"

    evap_in_file_source.data['file_contents'][0]=data_numpyarray
    # get just the name of the file without the extension 
    evap_in_file_source.data['file_name'][0]=evap_in_file_source.data['file_name'][0].split(".")[0]
    # write the file name
    evap_filename.text=evap_in_file_source.data['file_name'][0]

evap_in_file_source.on_change('data', evap_file_callback)

evap_upload_button = Button(label="Upload-Distributed Evapotranspiration", button_type="success", width = 230,height=20)
evap_upload_button.callback = CustomJS(args=dict(file_source=evap_in_file_source), code =java_functions.javaupload())

"""evap figure """
# flip the array as the column datasource already flip it 
evap_in_plot_source = ColumnDataSource(data=dict(image=[np.flipud(evap_in_file_source.data['file_contents'][0][:, :, 0])]))
# color bar
color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(evap_in_file_source.data['file_contents'][0]),
                            high=np.nanmax(evap_in_file_source.data['file_contents'][0]), nan_color="white")
color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                      border_line_color=None, location=(0,0)) #label_standoff=12
# figure
evap_input_plot= figure( x_range=(0, np.shape(evap_in_file_source.data['file_contents'][0])[1]), y_range=(0, np.shape(evap_in_file_source.data['file_contents'][0])[0]),
            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
evap_input_plot.image(image='image', x=0, y=0, dw=np.shape(evap_in_file_source.data['file_contents'][0])[1], dh=np.shape(evap_in_file_source.data['file_contents'][0])[0],
         source=evap_in_plot_source ,color_mapper=color_mapper)
evap_input_plot.add_layout(color_bar, 'right')

# TODO change the tag of the slider to day, hour, month and year 
evap_slider= Slider(title="Evapotranspiration index ", start=0, end=(data_length-1), value=0, step=1, )#title="Frame"
evap_date = TextInput(value = str(date_ds.data['date'][evap_slider.value]), title = 'date:',
                              width=60)
#
def update_evap(attr, old, new):
    # update the plot datasource
    evap_in_plot_source.data = dict(image=[np.flipud(evap_in_file_source.data['file_contents'][0][:, :, evap_slider.value])])
    # update the date text 
    evap_date.value=str(date_ds.data['date'][evap_slider.value])

evap_slider.on_change('value', update_evap)



### precipitation
prec_filename= Div(text = 'precipitation file',width=120, height=20)
prec_message=Div(text=" upload your data ", width=500, height=10)
# create a data source with the name and content stored in a dict
prec_in_file_source = ColumnDataSource({'file_contents':[0], 'file_name':["anyname"]})
prec_in_file_source.data['file_contents'][0]=sp_prec_c

def file_callback(attr,old,new):
#    print ('filename:', file_source.data['file_name'])
#    print ('filecontent:', file_source.data['file_contents'])
    # file_source.data['file_contents'] is a list of length 1 to access it
    # file_source.data['file_contents'][0]
    raw_contents = prec_in_file_source .data['file_contents'][0]
#    raw_contents = file_source.data['file_contents']
    # remove the prefix that JS adds
    # java will read data with a prefix data:text/plain;/plain,rtsert5q55878qaer 
    # split it with the first comma
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
    # now file_contents is exactly the shape of the data in the file to read it use stringIO
    # file_contents is a  string
    file_io = StringIO.StringIO(file_contents)
    # text
#    df = np.loadtxt(file_io, usecols=0)
    # Excel
#    df = pd.read_excel(raw_contents)
#    print(df[0])
#    print ("file contents:")
    # numpy array 
    data_numpyarray=np.load(file_io)
    # validation of the length of time series
#    print(np.shape(data_numpyarray)[2])
#    print(len(date_ds.data['date']))
    
    if len(date_ds.data['date']) != np.shape(data_numpyarray)[2]:
        prec_message.text=" precipitation data lenght does not match with star, end date and time step"
    else:
        prec_message.text=" date matches with length of precipitation input data"

     # file_source.data['file_contents']=(u'file_contents', 13350)
    prec_in_file_source.data['file_contents'][0]=data_numpyarray
#    print(prec_in_file_source.data['file_contents'][0][0,:,:])
    # get just the name of the file without the extension 
    prec_in_file_source.data['file_name'][0]=prec_in_file_source.data['file_name'][0].split(".")[0]
    # write the file name
    prec_filename.text=prec_in_file_source.data['file_name'][0]

prec_in_file_source .on_change('data', file_callback)

prec_upload_button = Button(label="Upload-Distributed Precipitation", button_type="success", width = 230,height=20)
prec_upload_button.callback = CustomJS(args=dict(file_source=prec_in_file_source), code =java_functions.javaupload())

"""prec figure """
# flip the array as the column datasource already flip it 
prec_in_plot_source = ColumnDataSource(data=dict(image=[np.flipud(prec_in_file_source.data['file_contents'][0][:, :, 0])]))
# color bar
color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(prec_in_file_source.data['file_contents'][0]),
                            high=np.nanmax(prec_in_file_source.data['file_contents'][0]), nan_color="white")
color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                      border_line_color=None, location=(0,0)) #label_standoff=12
# figure
precipitation_input_plot= figure( x_range=(0, np.shape(prec_in_file_source.data['file_contents'][0])[1]), y_range=(0, np.shape(prec_in_file_source.data['file_contents'][0])[0]),
            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
precipitation_input_plot.image(image='image', x=0, y=0, dw=np.shape(prec_in_file_source.data['file_contents'][0])[1], dh=np.shape(prec_in_file_source.data['file_contents'][0])[0],
         source=prec_in_plot_source ,color_mapper=color_mapper)
precipitation_input_plot.add_layout(color_bar, 'right')

# TODO change the tag of the slider to day, hour, month and year 
precipitation_slider= Slider(title="precipitation index ", start=0, end=(data_length-1), value=0, step=1, )#title="Frame"
precipitation_date = TextInput(value = str(date_ds.data['date'][precipitation_slider.value]), title = 'date:',
                              width=60) #, css_classes=['customTextInput']

#show(precipitation_date)

def update_precipitation(attr, old, new):
    # update the plot datasource
    prec_in_plot_source.data = dict(image=[np.flipud(prec_in_file_source.data['file_contents'][0][:, :, precipitation_slider.value])])
    # update the date text 
    precipitation_date.value=str(date_ds.data['date'][precipitation_slider.value])
    # TODO update the color bar
    # add commands her to update the color bar 

precipitation_slider.on_change('value', update_precipitation)

### run the model
# message

result_ds= ColumnDataSource({'Quz_dist':[0], 'Qout':[0]})
result_ds.data['Quz_dist'][0]=sp_quz_4km

performance = Div(text=" ")


def run_Distributed_model():
    performance.text = str("<h2>processing...<h2>")    
    sp_temp_c= temp_in_file_source.data['file_contents'][0]
    sp_et_c= evap_in_file_source.data['file_contents'][0]
    sp_prec_c=prec_in_file_source .data['file_contents'][0]
    # read values of parameters 
    
    # data validation
    
#    _area = float(w_area.value)
#    pars222222 = [_k1, _k2, _k3, _k4, _d1, _d2, _s1, _s2]
#    extra_pars22222 = [_dt, _area]
    
    performance_Q={}
#    calc_Q=pd.DataFrame(index=index)
#    q_tot, st , q_uz_routed, q_lz,_
    result_ds.data['Qout'][0],_,result_ds.data['Quz_dist'][0],_=Wrapper.Dist_model_lake(lake_data_A,
                                 extra_pars,curve,lakecell,DEM,flow_acc_table,flow_acc,sp_prec_c,sp_et_c,
                                 sp_temp_c, pars,kub,klb,jiboa_initial=jiboa_initial,
                                 lake_initial=lake_initial,ll_temp=None, q_0=None)
    print("calculation done")
    # Calculate model performance
    performance.text = str("<h2>calculating model performance..<h2>")
    
    WS={} #------------------------------------------------------------------------
    WS['type']=1 #------------------------------------------------------------------------
    WS['N']=3 #------------------------------------------------------------------------
    
    performance_Q['c_error_hf'] = PerformanceCriteria.rmseHF(lake_data['Q'],result_ds.data['Qout'][0],WS['type'],WS['N'],0.75)#------------------------------------------------------------------------
    performance_Q['c_error_lf']=PerformanceCriteria.rmseLF(lake_data['Q'],result_ds.data['Qout'][0],WS['type'],WS['N'],0.75)#------------------------------------------------------------------------
    performance_Q['c_nsehf']=PerformanceCriteria.nse(lake_data_A[:,-1],result_ds.data['Qout'][0])#------------------------------------------------------------------------
    performance_Q['c_rmse']=PerformanceCriteria.rmse(lake_data_A[:,-1],result_ds.data['Qout'][0])#------------------------------------------------------------------------
    performance_Q['c_nself']=PerformanceCriteria.nse(np.log(lake_data_A[:,-1]),np.log(result_ds.data['Qout'][0]))#------------------------------------------------------------------------
    performance_Q['c_KGE']=PerformanceCriteria.KGE(lake_data_A[:,-1],result_ds.data['Qout'][0])#------------------------------------------------------------------------
    performance_Q['c_wb']=PerformanceCriteria.WB(lake_data_A[:,-1],result_ds.data['Qout'][0])#------------------------------------------------------------------------
    
    # update data source
    ds_sim.data = (dict(q_sim = result_ds.data['Qout'][0].tolist(), ds_time = index))

    performance.text = str("<h2>Model perfomance(RMSE) is %s<h2>" %round(performance_Q['c_rmse'], 3))
    
# assign buttons
run_model_button.on_click(run_Distributed_model)


"""results figure """
# flip the array as the column datasource already flip it 
Quz_dist_in_plot_source= ColumnDataSource(data=dict(image=[np.flipud(result_ds.data['Quz_dist'][0][:, :, 0])]))
# color bar
color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(result_ds.data['Quz_dist'][0]),
                            high=np.nanmax(result_ds.data['Quz_dist'][0]), nan_color="white")
color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                      border_line_color=None, location=(0,0)) #label_standoff=12
# figure
Quz_dist_plot= figure( x_range=(0, np.shape(result_ds.data['Quz_dist'][0])[1]), y_range=(0, np.shape(result_ds.data['Quz_dist'][0])[0]),
            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
Quz_dist_plot.image(image='image', x=0, y=0, dw=np.shape(result_ds.data['Quz_dist'][0])[1], dh=np.shape(result_ds.data['Quz_dist'][0])[0],
         source=Quz_dist_in_plot_source ,color_mapper=color_mapper)
Quz_dist_plot.add_layout(color_bar, 'right')

# TODO change the tag of the slider to day, hour, month and year 
Quz_dist_slider= Slider(title="Runoff index ", start=0, end=(data_length-1), value=0, step=1, )#title="Frame"
Quz_dist_date = TextInput(value = str(date_ds.data['date'][Quz_dist_slider.value]), title = 'date:',
                              width=60)
#
def update_Quz_dist(attr, old, new):
    # update the plot datasource
    Quz_dist_in_plot_source.data = dict(image=[np.flipud(result_ds.data['Quz_dist'][0][:, :, Quz_dist_slider.value])])
    # update the date text 
    Quz_dist_date.value=str(date_ds.data['date'][Quz_dist_slider.value])

Quz_dist_slider.on_change('value', update_Quz_dist)

Main_title1 = Div(text="<h1  style=color:DodgerBlue;"
               "font-size:50px;font-family:comic sans ms;margin: auto;border: 3px solid green;"
               "width=500px;text-align: center;> Model setup<h1>", width=500) # ,css_classes=['customTextInput']
#Main_title1 = Div(text="<h55>Model setup</h55>" ) #,css_classes=['h55']

# TODO button to upload the parameters
# TODO graph to plot the parameters
# show the GUI
# widget boxes
inputdata_text=Div(text="<h2>Input Data<h2>", width=600, height=30)
resultdata_text=Div(text="<h2>Result<h2>", width=600, height=30)
#wb2 = widgetbox(check_text,run_model_button ,check_button,
#                performance, height = height)
#run_model_wb2 =row(prec_upload_button, width = 100,height=80)# widgetbox()#,w_area, height = height) 
empty_div2=Div(text=" ", width=200, height=20)
empty_div3=Div(text=" ", width=50, height=20)
empty_div4=Div(text=" ", width=150, height=20)

empty_div5=Div(text=" ", width=0, height=20)
empty_div6=Div(text=" ", width=200, height=20)
#empty_div7=Div(text="mostafa ",css_classes=['templates/styles.css']) # #, width=200, height=20
#show(empty_div7)
# make a grid
layout_runmodel=  layout( [[Main_title1], #gridplot
                           [dates_div],
                           [Time_step],
                           [row([startdate,empty_div1,endtdate])],
                           
                           [inputdata_text],
                           [prec_upload_button,prec_filename,empty_div2,evap_upload_button,evap_filename],
                           [prec_message,empty_div3,evap_message],
                           [row(precipitation_input_plot,evap_input_plot)],
                           [row([precipitation_slider,precipitation_date,empty_div4,evap_slider,evap_date])],
                           [temp_upload_button,temp_filename],
                           [temp_message],
                           [temp_input_plot],#,DEM_upload_button
                           [row([temp_slider,temp_date])],
                           [plot_sim, plot_evap],
                           [run_model_button],
#                           
                           [resultdata_text],
                           [Quz_dist_plot,plot_qsim],
                           [Quz_dist_slider,Quz_dist_date],
#                           [wb2],#, wb2
#                           [inputdata_title ],
                           ] )

run_model = Panel(child=layout_runmodel, title="Run the model")
''' ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================'''

#%% Input data
# default dates

s=dt.datetime(2011,12,31,17,00,00)
e=dt.datetime(2016,04,16,10,00,00)
date_tab2_ds=ColumnDataSource(data=dict(date=[]))

Time_step_in=Select(title="Time Step", value="15 min", options=["15 min","Hourly", "Daily", "weekly"])
startdate_in=TextInput(title="Start date:",value=str(s), width=100, height=20)
endtdate_in=TextInput(title="End date:",value=str(e), width=100, height=20)

empty_div1=Div(text=" ", width=100, height=20)
messages_div=Div(text="<h5 style=color:#cb2135;"
               "font-size:30px;font-family:comic sans ms;"
               "background-color: #cb2135 !important;>"
               "  <h5>",
               width = 300, height=20)

### station corrdinates 
#data=pd.read_csv(data_file+"station coordinates_8.csv") #,delimiter=" "
stations_coor_ds= ColumnDataSource({'file_contents':[0],'file_name':[0],'lat':[0],'long':[0],'name':[0]})
stations_coord_fname=Div(text="coordinates file", width=200)

station_list=Select(title="station list", value=" upload you data", options=["upload you data"])
station_plot_ds = ColumnDataSource(dict(ds_time = [0], st_values = [0]))

def stations_coord_f_callback(attr,old,new):
    raw_contents = stations_coor_ds.data['file_contents'][0]
    # free the memory 
    del stations_coor_ds.data['file_contents'][0]    
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
    # now file_contents is exactly the shape of the data in the file to read it use stringIO
    # file_contents is a  string
    file_io = StringIO.StringIO(file_contents)
    data=pd.read_csv(file_io)
    
    stations_coor_ds.data['name'][0]=data['name'].tolist()
    stations_coor_ds.data['lat'][0]=data['lat'].tolist()
    stations_coor_ds.data['long'][0]=data['long'].tolist()
    # write the file name
    stations_coord_fname.text=stations_coor_ds.data['file_name'][0]
    # update the select widget
    station_list.options = stations_coor_ds.data['name'][0]
    # recreate the gmap
#    layout_inputdata.children.remove(layout_inputdata.children[7])
    layout_inputdata.children[1].children.remove(layout_inputdata.children[1].children[-1])
    
    inputs_graph = GMapPlot(api_key="AIzaSyC2ThwIKXr03ENAOg1Etbfo0FoQUAs6_tI",
             plot_width=700, plot_height=500, x_range = Range1d(), y_range = Range1d(), 
             map_options=GMapOptions(lat=np.mean(stations_coor_ds.data['lat'][0]),
                                     lng=np.mean(stations_coor_ds.data['long'][0]),
             zoom=11,map_type="satellite"), toolbar_location="right") #border_fill = '#130f30')
    
    st_gmap_ds=ColumnDataSource(data=dict(name=stations_coor_ds.data['name'][0],
                                          x=stations_coor_ds.data['long'][0],
                                          y=stations_coor_ds.data['lat'][0]))
    
    inputs_graph_glyph =Circle(x="x", y="y",name="name", fill_color="#3D59AB", #,line_color="#3288bd"
                                   line_width=3,size=20)
    inputs_graph_glyph = inputs_graph.add_glyph(st_gmap_ds, inputs_graph_glyph )
    rainfall_station_hover = HoverTool(renderers=[inputs_graph_glyph ])
    rainfall_station_hover.tooltips = OrderedDict([
            ('Station Name:','@name'),
            ("(long:,lat:)", "($x, $y)"),
             ])
    inputs_graph.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(),ResetTool(),rainfall_station_hover)

#    show(inputs_graph)
#    layout_inputdata.children.append(inputs_graph)
#    layout_inputdata.children.insert(7,inputs_graph)
    layout_inputdata.children[1].children.append(inputs_graph)

stations_coor_ds.on_change('data', stations_coord_f_callback)
stations_coord_upload_button = Button(label="Upload-stations coordinates", button_type="success", width = 230,height=20)

stations_coord_upload_button.callback = CustomJS(args=dict(file_source=stations_coor_ds), code =java_functions.upload_stations())


### station values 
#data=pd.read_csv(data_file+"stations8.txt") #,delimiter=" "
stations_values_ds= ColumnDataSource({'file_contents':[0],'file_name':[0],})
stations_values_fname=Div(text="station valeus ", width=200)

def stations_values_callback(attr,old,new):
    raw_contents = stations_values_ds.data['file_contents'][0]
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
    # now file_contents is exactly the shape of the data in the file to read it use stringIO
    # file_contents is a  string
    file_io = StringIO.StringIO(file_contents)
    data=pd.read_csv(file_io)
    # aggregate
#    data.index=dates
#    data=data.resample('H').sum()
    data=data.as_matrix()
    stations_values_ds.data['file_contents'][0]=data
    #1- write the file name
    stations_values_fname.text=stations_values_ds.data['file_name'][0]
    
    #2- check the length of the time series with the date
    s=pf.changetext2time(startdate_in.value)
    e=pf.changetext2time(endtdate_in.value)
    
    if Time_step_in.value== "15 min":
        freq="15Min"
    if Time_step_in.value== "Hourly":
        freq="1H"
    elif Time_step_in.value== "Daily":
        freq="1D"
    elif Time_step_in.value== "weekly":
        freq="1W"

    dates=pd.date_range(startdate_in.value,endtdate_in.value,freq=freq)#.tolist()
    date_tab2_ds.data['date']=dates
    # check if the length of the times series (start and end date) matches with 
    # the length of the station readings uploaded by upload readings
    if len(dates) != len(stations_values_ds.data['file_contents'][0]):
        messages_div.text=("length of time series between dates"+str(len(dates))+
              " does not match with the length of the data in the file"+str(len(stations_values_ds.data['file_contents'][0])))

    #3- update the precipitation plot with the plot of the first station 
    station_plot_ds.data = dict(ds_time = dates, 
                                st_values = stations_values_ds.data['file_contents'][0][:,0])
    
    station_plot.line(x = 'ds_time', y = 'st_values', source = station_plot_ds, 
                      color="#27408B") # ,legend=stations_coor_ds.data['name'][0][0]
    
    station_plot.x_range=Range1d(start=s,end=e)
    station_plot.yaxis.axis_label = "Station value"
    station_plot.xaxis.axis_label = "Dates"
    # display the cal_dist button
    layout_inputdata.children[1].children[0].children.append(row(cal_dist,gen_message))

    

stations_values_ds.on_change('data', stations_values_callback)

stations_values_upload_button = Button(label="Upload-Stations readings", 
                                       button_type="success", 
                                       width = 230,height=20)

stations_values_upload_button.callback = CustomJS(args=dict(file_source=stations_values_ds), 
                                                  code =java_functions.javaupload())

### plot values

station_plot = figure(title="Stations",width=width2, height=500, 
                         toolbar_location = "right",x_axis_type = "datetime")

station_plot.line(x = 'ds_time', y = 'st_values', 
                  source = station_plot_ds, 
                  color="firebrick",) # legend='precipitation'

station_plot.yaxis.axis_label = "Discharge [m3/s]"
station_plot.xaxis.axis_label = "Dates"
station_plot.x_range=Range1d(start=pf.changetext2time(startdate_in.value),
                             end=pf.changetext2time(endtdate_in.value))


def plot_st_callback(attr,old,new):
    
    # check if the user has already uploaded the station values or not 
    if np.size(stations_values_ds.data['file_contents'][0])==1:
        # TODO java script popup message 
#        CustomJS(code ="""alert ("please upload your stations data") """)
        messages_div.text="<h5 style=color:#cb2135;font-size:30px;font-family:comic sans ms;background-color: #cb2135 !important;>please upload station readings  <h5>"
    
    # update the precipitation plot based on the selected station from the select button 
    # get the order of the selected station on the list
    selected_ind=station_list.options.index(station_list.value)
    station_plot_ds.data=dict(ds_time = date_tab2_ds.data['date'], st_values = stations_values_ds.data['file_contents'][0][:,selected_ind])

    # TODO update the legend do delete the previous station name and show only the current station name
    station_plot.line(x = 'ds_time', y = 'st_values', source = station_plot_ds, color="#27408B") #, legend=[stations_coor_ds.data['name'][0][selected_ind]]
    station_plot.x_range=Range1d(start=s,end=e)
    station_plot.yaxis.axis_label = "Station values "
    station_plot.xaxis.axis_label = "Dates"
#    station_plot.legend #=Legend()


station_list.on_change("value",plot_st_callback)

# plot the stations in gmap    
inputs_graph = GMapPlot(api_key="AIzaSyC2ThwIKXr03ENAOg1Etbfo0FoQUAs6_tI",
             plot_width=700, plot_height=500, x_range = Range1d(), y_range = Range1d(), 
             map_options=GMapOptions(lat=30.023, lng=31.19, zoom=1,map_type="satellite"), #
             toolbar_location="right") #border_fill = '#130f30')

inputs_graph.title = Title()
inputs_graph.title.text="Station Location"
inputs_graph.add_tools(PanTool(), WheelZoomTool(),ResetTool(), CrosshairTool())

#show(inputs_graph )
### calculation

# TODO button to upload the raster shapefile

"""generated data plot  """
## TODO change the tag of the slider to day, hour, month and year 
gen_message=Div(text=" Generate your distributed data", width=200)
gendata_slider= Slider(title="index ", start=0, end=(len(date_tab2_ds.data['date'])-1), value=0, step=1, )#title="Frame"
#gendata_slider= Slider(title="index ", start=0, end=0, value=0, step=1, width=400)#title="Frame"
#gendata_date = TextInput(value = str(date_tab2_ds.data['date'][0]), title = 'date:',width=60) #, css_classes=['customTextInput']
gendata_date = TextInput(value = " ", title = 'date:',width=60) #, css_classes=['customTextInput']

gendata_ds = ColumnDataSource(data=dict(dist_data=[]))
gendata_plot_ds= ColumnDataSource(data=dict(image=[]))

# create a color ramp
colors=['#EFF2FA','#DBDCF9','#C7C7F9','#B4B1F9','#A09CF9','#8D86F9','#7971F8','#655BF8',
        '#5246F8','#3E30F8','#2B1BF8']
poly_ds=ColumnDataSource(data=dict(x=[],y=[],value=[],color=[]))

    
colors_ds=ColumnDataSource(data=dict(high=[0],low=[0]))
inputs_graph2 = GMapPlot(api_key="AIzaSyC2ThwIKXr03ENAOg1Etbfo0FoQUAs6_tI",
             plot_width=700, plot_height=500, x_range = Range1d(), y_range = Range1d(),)

poly_patches = Patches(xs="x", ys="y", fill_color="color",
                      fill_alpha=0.6, line_color="black", line_width=2)
    
def generate_dist_data():
    gen_message.text="generating data started "
    # change the coordinate web mercator 
    x,y=plf.reproject_points(stations_coor_ds.data['lat'][0],stations_coor_ds.data['long'][0])
    coordinates=dict(x=x,y=y)
    # read the uploaded dem    
    inputspath="HBV_distributed/static/input_data/DEM/"
    #source
    src = gdal.Open(inputspath+"dem_4km.tif")
    # change coordinate system of the dem to web mercator
    dst=plf.reproject_dataset(src)
    #    stations_values_ds.data['file_contents'][0]=data.as_matrix()
    gendata_ds.data['dist_data']=st.ISDW(dst,coordinates,stations_values_ds.data['file_contents'][0])
        
    print("Generating the data finished")
    gen_message.text="Generating the data finished"
    
    src_array = src.ReadAsArray()
    no_val=src.GetRasterBand(1).GetNoDataValue()
    
    shape_base_dem = src.ReadAsArray().shape
    
    corner1=np.float32(np.ones((shape_base_dem[0]+1,shape_base_dem[1]+1,2))*np.nan)
    gt = src.GetGeoTransform()
    # get coordinates of all edges 
    for i in range(shape_base_dem[0]+1): # iteration by row
        for j in range(shape_base_dem[1]+1):# iteration by column
    #        if raster_array[i,j] != no_val:
            # gt, column , row
            corner1[i,j,0],corner1[i,j,1]=gdal.ApplyGeoTransform(gt,j,i) # bottom right corner
    # store coordinates as tuples
    coords=dict()
    for i in range(shape_base_dem[0]): # iteration by row
        for j in range(shape_base_dem[1]):# iteration by column
            if src_array [i,j] != no_val:
                coords[str(i)+","+str(j)]=[]
                # store upper left corner
                coords[str(i)+","+str(j)].append((corner1[i,j,0],corner1[i,j,1]))
                # store upper right corner
                coords[str(i)+","+str(j)].append((corner1[i,j+1,0],corner1[i,j+1,1]))
                # store bottom right corner
                coords[str(i)+","+str(j)].append((corner1[i+1,j+1,0],corner1[i+1,j+1,1]))
                # store upper corner
                coords[str(i)+","+str(j)].append((corner1[i+1,j,0],corner1[i+1,j,1]))
    
    proj=src.GetProjection()
    src_epsg=osr.SpatialReference(wkt=proj)
    # create a geodataframe with index as cell indeces
    poly_df=gpd.GeoDataFrame(index=coords.keys())
    # set the coordinate system of the geo data frame as the dem
    poly_df.crs=from_epsg(src_epsg.GetAttrValue('AUTHORITY',1))
    poly_df['coords']=""
    poly_df["geometry"]=""
    # store  the coordinates and create the polygons     
    for i in range(len(coords.keys())):
        poly_df.loc[coords.keys()[i],'coords']=coords[coords.keys()[i]]
    #    poly_df.loc[coords.keys()[i],"geometry"]=plf.create_polygon(coords[coords.keys()[i]])
        poly_df.loc[coords.keys()[i],"geometry"]=Polygon(coords[coords.keys()[i]])
    
    # store cell index in a column 'cells'
    poly_df['cells']=poly_df.index
    poly_df.index=[i for i in range(len(poly_df))]
    # convert the CRS to geographic 
    poly_df.loc[0:len(poly_df),'geometry']=poly_df['geometry'].to_crs(epsg=4326)
    
    layout_inputdata.children[1].children.remove(layout_inputdata.children[1].children[-1]) # ___________________________________________________________
    
    # plot the figure 
    inputs_graph2 = GMapPlot(api_key="AIzaSyC2ThwIKXr03ENAOg1Etbfo0FoQUAs6_tI",
             plot_width=700, plot_height=500, x_range = Range1d(), y_range = Range1d(), 
             map_options=GMapOptions(lat=np.mean(stations_coor_ds.data['lat'][0]),
                                     lng=np.mean(stations_coor_ds.data['long'][0]),
             zoom=11,map_type="satellite"), toolbar_location="right") #border_fill = '#130f30')
    
    st_gmap_ds=ColumnDataSource(data=dict(name=stations_coor_ds.data['name'][0],
                                          x=stations_coor_ds.data['long'][0],
                                          y=stations_coor_ds.data['lat'][0]))
    
    inputs_graph_glyph =Circle(x="x", y="y",name="name", fill_color="#3D59AB", #,line_color="#3288bd"
                                   line_width=3,size=20)
    inputs_graph_glyph = inputs_graph2.add_glyph(st_gmap_ds, inputs_graph_glyph )
    rainfall_station_hover = HoverTool(renderers=[inputs_graph_glyph ])
    rainfall_station_hover.tooltips = OrderedDict([
            ('Station Name:','@name'),
            ("(long:,lat:)", "($x, $y)"),
             ])
    inputs_graph2.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(),ResetTool(),)#rainfall_station_hover
    
    
    low=np.max([np.nanmin(gendata_ds.data['dist_data']),0])
    high=np.nanmax(gendata_ds.data['dist_data'])
#    colors_ds=ColumnDataSource(data=dict(low=[low],high=[high]))
    colors_ds.data['low']=[low]
    colors_ds.data['high']=[high]
    
    # get the coordinates of the edges with the gmap CRS
    poly_df1=plf.XY(poly_df)
    
    poly_df1['value']=""
    poly_df1['color']=""
    
    for i in range(len(poly_df1)):
        x,y=poly_df1['cells'][i].split(",")
        poly_df1['value'][i]=gendata_ds.data['dist_data'][:, :, 0][int(x),int(y)]
        # assign color based on the value
#        gendata_ds.data['dist_data'][gendata_ds.data['dist_data']==0]=np.nan
        poly_df1['color'][i]=colors[int(((poly_df1['value'][i]-low)/(high-low))*(len(colors)-0))]
    
    del poly_df1['coords'], poly_df1['geometry'] #poly_df1['cells']
      
    poly_ds.data['cells']=poly_df1['cells']
    poly_ds.data['x']=poly_df1['x']
    poly_ds.data['y']=poly_df1['y']
    poly_ds.data['value']=poly_df1['value']
    poly_ds.data['color']=poly_df1['color']
#     patch for the catchment polygon 
    poly_patches = Patches(xs="x", ys="y", fill_color="color",
                      fill_alpha=0.8, line_color="black", line_width=2)
    
    poly_glyph = inputs_graph2.add_glyph(poly_ds, poly_patches)
#    gendata_plot_ds.data['image']=[np.flipud(gendata_ds.data['dist_data'][:, :, 0])]
        
#    color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(gendata_ds.data['dist_data']),
#                                high=np.nanmax(gendata_ds.data['dist_data']), nan_color="white")
#    color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
#                          border_line_color=None, location=(0,0)) #label_standoff=12
##     figure
#    gendata_plot= figure( x_range=(0, np.shape(gendata_ds.data['dist_data'][:,:,0])[1]),
#                         y_range=(0, np.shape(gendata_ds.data['dist_data'][:,:,0])[0]),
#                toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
#    
#    gendata_plot.image(image='image', x=0, y=0, dw=np.shape(gendata_ds.data['dist_data'][:,:,0])[1],
#                       dh=np.shape(gendata_ds.data['dist_data'][:,:,0])[0],
#                       source=gendata_plot_ds, color_mapper=color_mapper)
    
#    gendata_plot.add_layout(color_bar, 'right')
    gendata_slider.end=len(date_tab2_ds.data['date'])-1
    layout_inputdata.children[1].children.append(inputs_graph2)
    layout_inputdata.children[2].children.append(row(empty_div21,gendata_slider,gendata_date))

#    layout_inputdata.children[3].children.append(layout([row(gendata_slider,gendata_date)]))

    
cal_dist= Button(label="Generate distributed data", button_type="success", width = 230,height=30,)
cal_dist.on_click(generate_dist_data)


def update_gendata(attr, old, new):
    # update the plot datasource
#    gendata_plot_ds.data['image']=[np.flipud(gendata_ds.data['dist_data'][:, :, gendata_slider.value])]

    for i in range(len(poly_ds.data['value'])): 
        x,y=poly_ds.data['cells'][i].split(",")
        poly_ds.data['value'][i]=gendata_ds.data['dist_data'][:, :, gendata_slider.value][int(x),int(y)]
        # assign color based on the value
        # logarithmic scale
#        color_index=pf.mycolor(poly_ds.data['value'][i],min_old,max_old,0,len(colors))
        color_index=pf.mycolor(poly_ds.data['value'][i],colors_ds.data['low'][0],
                               colors_ds.data['high'][0],0,len(colors))
        poly_ds.data['color'][i] = colors[color_index]
        # linear scale
#        poly_ds.data['color'][i]=colors[int(((poly_ds.data['value'][i]-colors_ds.data['low'][0])/(colors_ds.data['high'][0]-colors_ds.data['low'][0]))*(len(colors)-0))]
        # calculate percentiles         
#        np.nanpercentile(np.extract(gendata_ds.data['dist_data'] != 0,gendata_ds.data['dist_data']),90)
        print(gendata_slider.value)
#    print(gendata_slider.value)    
    poly_ds.trigger('data', poly_ds.data, poly_ds.data)
#    inputs_graph2_glyhp=inputs_graph2.add_glyph(poly_ds, poly_patches)
#    inputs_graph2.set_select('add_glyph',inputs_graph2_glyhp)
    
#    inputs_graph2.update('add_glyph', inputs_graph2_glyhp)

    # update the date text 
    gendata_date.value=str(date_tab2_ds.data['date'][gendata_slider.value])
    # TODO update the color bar
gendata_slider.on_change('value', update_gendata)


# show the GUI
input_data_Description = Div(text=" <h3> This section is for generating distributed data"
                             "from simple statistical methods like inverse distance weighting"
                             " method (IDWM) and inverse inverse squared distance weighting method (ISDWM) "
                             "<h3>",width = 1100)

row1 = widgetbox(input_data_Description )

col1 = layout([Time_step_in],
              [row(startdate_in,empty_div1,endtdate_in)],
              [row(stations_coord_upload_button,stations_coord_fname) ],
              [row(stations_values_upload_button,stations_values_fname)],
              [messages_div],
              )
empty_div20=Div(text=" ", width=50, height=20)
empty_div21=Div(text=" ", width=200, height=20)

des_div_in2=Div(text="Description", width=500, height=30)
des_div_in3=Div(text="Input 3", width=500, height=30)


layout_inputdata =layout(children=[
#                    [indata_op],
                    [row1],
                    [col1 ,empty_div20 ,inputs_graph],
                    [row([station_list],sizing_mode="fixed")], #scale_height
                    [station_plot ],
                    ], sizing_mode="fixed")

#%% input 2
create_input="HBV_distributed/static/input_data/create_inputs/"
#source
src = gdal.Open(create_input+"dem_4km.tif")
    
#gis=Inputs(src)
#FD,elev=gis.FD_from_DEM()

layout_inputdata2 = layout(children=[
                                    [des_div_in2],
                                    ], sizing_mode="fixed")

#%% input 3
layout_inputdata3 = layout(children=[
                                [des_div_in3],
                                ], sizing_mode="fixed")

indata_op1 = Panel(child=layout_inputdata , title="Model Input 1")
indata_op2 = Panel(child=layout_inputdata2 , title="Model Input 2")
indata_op3 = Panel(child=layout_inputdata3 , title="Model Input 3")

layout_inputdata_tot = Tabs(tabs=[indata_op1, indata_op2, indata_op3], sizing_mode="fixed")

input_data = Panel(child = layout_inputdata_tot, title="Input Data")

'''
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
'''
#%% input 2
#input2_intro=Div(text="this section is for generating the second input to the model")
#layout_input2= layout ( [ 
#                     [input2_intro],
#                   ] )
#
#input2_model = Panel(child=layout_input2, title="Calibrate the model")
#

#%% Calibrate the model

# text input
hbv_parameters = Div(text=" <h3> HBV Parameters \n <h3>")
upperbound = Div(text="<h3> Upper Bound \n <h3>")
lowerbound = Div(text="<h3> Lower Bound \n <h3>")

w_ltt_u = TextInput(value = '0.5', title = 'Lower temperature threshold (ltt C)')
w_ltt_l = TextInput(value = '0.5', title = 'Lower temperature threshold (ltt C)')
w_utt_u = TextInput(value = '0.2', title = 'Upper temperature threshold (Utt C)')
w_utt_l = TextInput(value = '0.2', title = 'Upper temperature threshold (Utt C)')
w_sfcf_u = TextInput(value = '0.01', title = 'Snowfall correction factor (Sfcf)')
w_sfcf_l = TextInput(value = '0.01', title = 'Snowfall correction factor (Sfcf)')
w_rfcf_u = TextInput(value = '0.01', title = 'Rainfall correction factor (Rfcf)')
w_rfcf_l = TextInput(value = '0.01', title = 'Rainfall correction factor (Rfcf)')

w_cfmax_u = TextInput(value = '10.0', title = 'Day degree factor (Cfmax mm/ c.day)')
w_cfmax_l = TextInput(value = '10.0', title = 'Day degree factor (Cfmax mm/ c.day)')
w_ttm_u = TextInput(value = '20.0', title = 'Temperature threshold for melting (Ttm C)')
w_ttm_l = TextInput(value = '20.0', title = 'Temperature threshold for melting (Ttm C)')
w_whc_u = TextInput(value = '1.0', title = 'Holding water capacity (Whc)')
w_whc_l = TextInput(value = '1.0', title = 'Holding water capacity (Whc)')
w_cfr_u = TextInput(value = '1.0', title = 'Refreezing factor (Cfr)')
w_cfr_l = TextInput(value = '1.0', title = 'Refreezing factor (Cfr)')

w_fc_u = TextInput(value = '0.1', title = 'Maximum soil moisture storage (FC mm)')
w_fc_l = TextInput(value = '0.1', title = 'Maximum soil moisture storage (FC mm)')

w_beta_u = TextInput(value = '0.1', title = 'Nonlinear runoff parameter (Beta)')
w_beta_l = TextInput(value = '0.1', title = 'Nonlinear runoff parameter (Beta)')

w_ecorr_u = TextInput(value = '0.1', title = 'Evapotranspiration correction factor ()')
w_ecorr_l = TextInput(value = '0.1', title = 'Evapotranspiration correction factor ()')

w_lp_u = TextInput(value = '0.1', title = 'Limit for potential evaporation (LP %)')
w_lp_l = TextInput(value = '0.1', title = 'Limit for potential evaporation (LP %)')

w_cflux_u = TextInput(value = '0.1', title = 'Maximum capillary rate (Cflux mm/h)')
w_cflux_l = TextInput(value = '0.1', title = 'Maximum capillary rate (Cflux mm/h)')

w_k_u = TextInput(value = '0.1', title = 'Upper storage coefficient (K 1/h)')
w_k_l = TextInput(value = '0.1', title = 'Upper storage coefficient (K 1/h)')

w_k1_u = TextInput(value = '0.1', title = 'Lower storage coefficient (K1 1/h)')
w_k1_l = TextInput(value = '0.1', title = 'Lower storage coefficient (K1 1/h)')

w_alpha_u = TextInput(value = '0.1', title = 'Nonlinear response parameter (alpha)')
w_alpha_l = TextInput(value = '0.1', title = 'Nonlinear response parameter (alpha)')

w_perc_u = TextInput(value = '0.1', title = 'Percolation rate (Perc mm/h)')
w_perc_l = TextInput(value = '0.1', title = 'Percolation rate (Perc mm/h)')

w_clake_u = TextInput(value = '0.1', title = 'Lake correction factor (Clake)')
w_clake_l = TextInput(value = '0.1', title = 'Lake correction factor (Clake)')

w_krouting_u = TextInput(value = '0.1', title = 'Muskingum travelling time coefficient (K h)')
w_krouting_l = TextInput(value = '0.1', title = 'Muskingum travelling time coefficient (K h)')

w_xrouting_u = TextInput(value = '0.1', title = 'Muskingum weighting coefficient (X)')
w_xrouting_l = TextInput(value = '0.1', title = 'Muskingum weighting coefficient (X)')





## make the widgets
calibration_tab_intro = Div(text="<h3>This section of for calibrating the model Using Harmony Search "
                            "Algorithm <h3>",width=1100)
wb_calibrate_u = widgetbox(upperbound ,w_utt_u,w_ltt_u,w_sfcf_u,w_rfcf_u,w_cfmax_u,w_ttm_u,w_whc_u
                           ,w_cfr_u,w_fc_u,w_beta_u,w_ecorr_u,w_lp_u,w_cflux_u,w_k_u,
                           w_k1_u,w_alpha_u,w_perc_u,w_clake_u,w_krouting_u,w_xrouting_u)
wb_calibrate_l = widgetbox(lowerbound ,w_utt_l,w_ltt_l,w_sfcf_l,w_rfcf_l,w_cfmax_l,w_ttm_l,
                           w_whc_l,w_cfr_l,w_fc_l,w_beta_l,w_ecorr_l,w_lp_l,w_cflux_l
                           ,w_k_l,w_k1_l,w_alpha_l,w_perc_l,w_clake_l,w_krouting_l,w_xrouting_l)


layout_calibration = layout ( [ [calibration_tab_intro],
                                   [hbv_parameters],
                                   [wb_calibrate_u,wb_calibrate_l]
#	                   [wb_fsf, wb2_fsf, col2_fsf] ,
#                       [thu, thd, export_fsf]
                   ] )

calibrate_model = Panel(child=layout_calibration, title="Calibrate the model")
'''
# ==========================================================================================================================================================
==========================================================================================================================================================
==========================================================================================================================================================
'''
#%% Home Page
home_div = Div(text="<h1 style=color:DodgerBlue;"
               "font-size:50px;font-family:comic sans ms >"
               "Conceptual Distributed Hydrological Modelling <h1>",
               width = 1100)


intro_div = Div(text="<h1>Introduction<h1>"
               "<h3>This Model Has been implemented as a part of Master thesis on "
               "Spatio-temporal simulation of catchment response based on dynamic weighting of hydrological models"
               "with a case study in Jiboa catchment in El Salvador"
               "<h3>",
               width = 1100)

#catchment_img= Div(text = "<img src='HBV_distributed/static/images/cover.jpg' "
#                "style=width:500px;height:500px;vertical-align: middle >" )

case_study_details = Div(text = "<h3> Case Study: El Salvador <h3>"
                 "River: Jiboa River"
                 "<br>")
# map 
# shape files links 
lumped_catch = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data'), "Lumped_catchment.shp")
rainfall_station_path = os.path.join(os.path.abspath('HBV_distributed/static/input_data/plot_data'), "rainfall_station.shp")
# read files
# catchment
boundary=gpd.read_file(lumped_catch)  # polygon geometry type
boundary['legend']="Catchment"
# stations
rainfall_station=gpd.read_file(rainfall_station_path)  # polygon geometry type
rainfall_station['legend']="Rainfall Station"
runoff_station=rainfall_station[rainfall_station["ESTACION"]=="Puente Viejo"]

# change the coordinate system to GCS
boundary['geometry']=boundary['geometry'].to_crs(epsg=4326)
rainfall_station['geometry']=rainfall_station['geometry'].to_crs(epsg=4326)
runoff_station['geometry']=runoff_station['geometry'].to_crs(epsg=4326)

# read the coordinates of the shapefiles 
boundary=plf.XY(boundary)
rainfall_station=plf.XY(rainfall_station)
runoff_station=plf.XY(runoff_station)
# colors 
boundary['colors']= ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a",]# "#ef3b2c", "#cb181d", "#99000d"] # reds

"""creating the map object """
# google map
catchmentmap_intro_page = GMapPlot(api_key="AIzaSyC2ThwIKXr03ENAOg1Etbfo0FoQUAs6_tI",
             plot_width=1000, plot_height=800, x_range = Range1d(), y_range = Range1d(), 
             map_options=GMapOptions(lat=13.6, lng=-88.957844, zoom=11,map_type="satellite"),
             toolbar_location="right",) #border_fill = '#130f30')
# normal plot
#catchmentmap_intro_page =figure(title="Case Study", #tools=TOOLS,
#         plot_width=650, plot_height=500, active_scroll = "wheel_zoom")
# Do not add grid line
#helsinki.grid.grid_line_color = None

"""catchment polygons  """
catchmentmap_df=boundary[['Basin_ID','Watershed',"Area_km2","x","y","colors","legend"]]
catchmentmap_dfsource=ColumnDataSource(data=catchmentmap_df)

#Add polygon grid and a legend for it
# patch for the normal graph
#grid=catchmentmap_intro_page.patches("x","y", source=catchmentmap_dfsource, name="grid", 
##               fill_color={"field":"color"},
#               fill_alpha=1.0, line_color="black", line_width=0.03, legend="label_pt")

# patch for the catchment polygon 
catchmentmap_intro_page_patches = Patches(xs="x", ys="y", fill_color="colors",
                  fill_alpha=0.1, line_color="black", line_width=2)
catchmentmap_intro_page_glyph = catchmentmap_intro_page.add_glyph(catchmentmap_dfsource, catchmentmap_intro_page_patches)
# TODO add legend
# tools
Polygon_hover = HoverTool(renderers=[catchmentmap_intro_page_glyph])
#Polygon_hover = catchmentmap_intro_page .select(dict(type=HoverTool))
Polygon_hover.tooltips = OrderedDict([
        ('Basin ID','@Basin_ID'),
        ('Catchment Name','@Watershed'),
        ("Area (km2)", "@Area_km2"),
        ("(long)", "$x,"),
         ("lat",(" $y"))
         ])
    
"""mark at the rainfall station"""
# don't but the rainfall_station instead it will make an error as it has geometry column
rainfall_station_source=ColumnDataSource(data=dict(x=rainfall_station.x, y=rainfall_station.y,
                                                   name=rainfall_station.ESTACION,
                                                   Department=rainfall_station.DEPARTAMEN,
                                                   Elevation=rainfall_station.DEM_projec))
                                                   
rainfall_circles_glyph =Circle(x="x", y="y",name="name", fill_color="#3D59AB", #,line_color="#3288bd"
                               line_width=3,size=20)
# TODO add legend
catchmentmap_intro_page_glyph = catchmentmap_intro_page.add_glyph(rainfall_station_source, rainfall_circles_glyph )

# hover  for rainfall gauges 
rainfall_station_hover = HoverTool(renderers=[catchmentmap_intro_page_glyph ])
rainfall_station_hover.tooltips = OrderedDict([
        ('Station Name:','@name'),
        ('Department Name:','@Department'),
        ("Elevation (m):", "@Elevation"),
        ("(long:,lat:)", "($x, $y)"),
         ])
catchmentmap_intro_page.add_tools(Polygon_hover,PanTool(),rainfall_station_hover, 
                                  WheelZoomTool(), BoxSelectTool())
# legend
catchmentmap_intro_page.legend.location="top_right"
catchmentmap_intro_page.legend.orientation = "vertical"


"""runoff station"""
runoff_station_source=ColumnDataSource(data=dict(x=runoff_station.x, y=runoff_station.y,
                                                   name=runoff_station.ESTACION,
                                                   Department=runoff_station.DEPARTAMEN,
                                                   Elevation=runoff_station.DEM_projec))

runoff_station_square_glypg= Square(x="x", y="y",name="name", fill_color="#FF34B3", #,line_color="#3288bd"
                               line_width=3,size=20)
catchmentmap_intro_page_glyph = catchmentmap_intro_page.add_glyph(runoff_station_source, runoff_station_square_glypg )

#show(catchmentmap_intro_page)


wb_home = widgetbox(home_div)

layout_home=layout ([ [wb_home],
          [intro_div],
          [catchmentmap_intro_page], 
          [case_study_details],
                      ])
hometab = Panel(child = layout_home, title = "HOME")
#%% creating tabs
tabs = Tabs(tabs=[ hometab, input_data, calibrate_model, run_model,])

curdoc().add_root(tabs)


