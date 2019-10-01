"""
main page
"""
import os
os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model")

import sys
sys.path.append("HBV_distributed/function")
#%% Library
import numpy as np
import pandas as pd
import datetime as dt
import gdal
from math import pi
import StringIO
import base64
import geopandas as gpd
from collections import OrderedDict
import pysal as ps
from datetime import datetime,date
#import time

# bokeh
from bokeh.layouts import layout, widgetbox, gridplot , column, row
from bokeh.models.widgets import Button, TextInput, Div, Tabs ,Slider#, RadioGroup
from bokeh.models.widgets import Panel #DataTable, DateFormatter, TableColumn, 
from bokeh.models import (ColumnDataSource, CustomJS, GMapPlot,GMapOptions, 
                          LinearAxis, Range1d, HoverTool, PanTool, WheelZoomTool,
                          ResetTool, SaveTool, BoxSelectTool, ColorBar,LogColorMapper,
                         # NumeralTickFormatter, #PrintfTickFormatter, #BoxSelectionOverlay 
                          Circle, Square) #, Slider,GeoJSONDataSource, PreviewSaveTool,

from bokeh.plotting import figure#, gmap
from bokeh.io import curdoc, show
#from bokeh.palettes import YlOrRd6 as palette
#from bokeh.palettes import RdYlGn10 as palette
from bokeh.models.glyphs import Patches #, Line, Circle

from bokeh.models.widgets import DateRangeSlider, DateFormatter, DataTable, TableColumn #,DatePicker, NumberFormatter
#from bokeh.resources import CDN
#from bokeh.embed import components, autoload_static, autoload_server
#from bokeh.layouts import row
#from bokeh.plotting import save

# functions
import DHBV_functions
import Wrapper
import Performance_criteria
import plotting_functions as plf

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

flow_acc_table=DHBV_functions.load_obj(data_file +"flow_acc_table")
flow_acc=np.load(data_file +'flow_acc.npy')

DEM = gdal.Open(data_file+"/DEM/"+"dem_4km.tif")
elev, no_val=DHBV_functions.get_raster_data(DEM)

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

jiboa_par,lake_par=DHBV_functions.par2d_lumpedK1(pars,DEM,12,13,kub,klb)

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
                           prec = prec,))
# calculated discharge 
q_sim=np.loadtxt(data_file +"Q4km.txt")[0:len(prec)]
ds_sim = ColumnDataSource(dict( q_sim = q_sim , ds_time = index))

# Create Data Table
#columns_sug = [
#        TableColumn(field='ds_time', title="Date", 
#                    formatter=DateFormatter(format = 'ddMyy')),
#        TableColumn(field="prec", title="Precipitation"),
#        TableColumn(field="snow", title="Snowfall"),
#        TableColumn(field="evap", title="Actual ET"),
#        TableColumn(field="q_rec", title="Recorded Discharge"),] 
#
#data_table = DataTable(source=ds_rec, 
#                       columns=columns_sug, 
#                       width=2*630,height=340)

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
w_button = Button(label = 'Run model', button_type = 'success' , width = 150)
calibrate_button = Button(label = 'Calibrate model', button_type = 'warning', width = 150)

# message
performance = Div(text=" ")
# define the update

def run_Distributed_model():
    performance.text = str("<h2>processing...<h2>")    
    # read values of parameters 
    
    # data validation
    
#    _area = float(w_area.value)
#    pars222222 = [_k1, _k2, _k3, _k4, _d1, _d2, _s1, _s2]
#    extra_pars22222 = [_dt, _area]
    
    performance_Q={}
    calc_Q=pd.DataFrame(index=index)
    
    calc_Q['Q'],_,_,_,_,_=Wrapper.Dist_model(lake_data_A,
                                 extra_pars,curve,lakecell,DEM,flow_acc_table,flow_acc,sp_prec_c,sp_et_c,
                                 sp_temp_c, pars,kub,klb,jiboa_initial=jiboa_initial,
                                 lake_initial=lake_initial,ll_temp=None, q_0=None)
    
    # Calculate model performance
    performance.text = str("<h2>calculating model performance..<h2>")
    
    WS={} #------------------------------------------------------------------------
    WS['type']=1 #------------------------------------------------------------------------
    WS['N']=3 #------------------------------------------------------------------------
    
    performance_Q['c_error_hf']=Performance_criteria.rmseHF(lake_data['Q'],calc_Q['Q'],WS['type'],WS['N'],0.75)#------------------------------------------------------------------------
    performance_Q['c_error_lf']=Performance_criteria.rmseLF(lake_data['Q'],calc_Q['Q'],WS['type'],WS['N'],0.75)#------------------------------------------------------------------------
    performance_Q['c_nsehf']=Performance_criteria.nse(lake_data_A[:,-1],calc_Q['Q'])#------------------------------------------------------------------------
    performance_Q['c_rmse']=Performance_criteria.rmse(lake_data_A[:,-1],calc_Q['Q'])#------------------------------------------------------------------------
    performance_Q['c_nself']=Performance_criteria.nse(np.log(lake_data_A[:,-1]),np.log(calc_Q['Q']))#------------------------------------------------------------------------
    performance_Q['c_KGE']=Performance_criteria.KGE(lake_data_A[:,-1],calc_Q['Q'])#------------------------------------------------------------------------
    performance_Q['c_wb']=Performance_criteria.WB(lake_data_A[:,-1],calc_Q['Q'])#------------------------------------------------------------------------
    
    # update data source
    ds_sim.data = (dict(q_sim = calc_Q['Q'].tolist(), ds_time = index))

    performance.text = str("<h2>Model perfomance(RMSE) is %s<h2>" %round(performance_Q['c_rmse'], 3))
    
# assign buttons
w_button.on_click(run_Distributed_model)

Main_title1 = Div(text="<h3>Model setup <h3>")
inputdata_title = Div(text="<h3>Input Data from file<h3>")
#tbl_label = Div(text="<h3>Table inputs<h3>")
#%% upload button 

check_text = TextInput(value = ' ', title = 'check area')
#prec_filename= TextInput(value = 'precipitation file',width=80)
prec_filename= Div(text = 'precipitation file',width=80)
# create a data source with the name and content stored in a dict
file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})
#file_source.data['file_contents'][0]=sp_quz_4km
def file_callback(attr,old,new):
#    print ('filename:', file_source.data['file_name'])
#    print ('filecontent:', file_source.data['file_contents'])
    raw_contents = file_source.data['file_contents'][0]
#    raw_contents = file_source.data['file_contents']
    # remove the prefix that JS adds
    # java will read data with a prefix data:text/plain;/plain, 
    # split it 
    prefix, b64_contents = raw_contents.split(",", 1)
    # decode the values from base64 to decimals
    file_contents = base64.b64decode(b64_contents)
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
#    print (type(df))
     # file_source.data['file_contents']=(u'file_contents', 13350)
    file_source.data['file_contents'][0]=data_numpyarray
    print(file_source.data['file_contents'][0][0,0,0])
    # write the file name
#    prec_filename.value=file_source.data['file_name'][0]
    prec_filename.text=file_source.data['file_name'][0]
    # update the precipitation input figure 
#    return df
file_source.on_change('data', file_callback)
upload_button = Button(label="Upload-Distributed Precipitation", button_type="success", width = 150)
# java script to be assigned to the callback of the button when clicked it 
# will be execute the function (file_callback) then change the value of the variable 
upload_button.callback = CustomJS(args=dict(file_source=file_source), code = """
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
""")
def check_function():
    check_text.value=str(file_source.data['file_contents'][0][0,0,0])

#%%
sp_quz_4km=np.load(data_file + "q_uz_c_4km.npy")

s=dt.datetime(2012,6,14,19,0,0)
e=dt.datetime(2013,12,23,0,0,0)

dates=pd.date_range(s,e,freq="1H")
#n_years=e.year-s.year
N=np.shape(sp_quz_4km)[2]



# flip the array as the column datasource already flip it 
source = ColumnDataSource(data=dict(image=[np.flipud(sp_quz_4km[:, :, 0])]))
# color bar
color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(sp_quz_4km),
                            high=np.nanmax(sp_quz_4km), nan_color="white")
color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                      border_line_color=None, location=(0,0)) #label_standoff=12
# figure
TOOLS="pan, box_zoom, reset, save, crosshair"
precipitation_input_plot= figure( x_range=(0, np.shape(sp_quz_4km)[1]), y_range=(0, np.shape(sp_quz_4km)[0]),
            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
precipitation_input_plot.image(image='image', x=0, y=0, dw=np.shape(sp_quz_4km)[1], dh=np.shape(sp_quz_4km)[0],
         source=source,color_mapper=color_mapper)
precipitation_input_plot.add_layout(color_bar, 'right')

# TODO change the tag of the slider to day, hour, month and year 
precipitation_slider= Slider(title="precipitation index ", start=0, end=(N-1), value=0, step=1, )#title="Frame"
precipitatin_date = TextInput(value = str(dates[precipitation_slider.value]), title = 'date:',
                              width=80)
#
def update_precipitation(attr, old, new):
    source.data = dict(image=[np.flipud(sp_quz_4km[:, :, precipitation_slider.value])])
    precipitatin_date.value=str(dates[precipitation_slider.value])

precipitation_slider.on_change('value', update_precipitation)
#%%    
check_button= Button(label="check", button_type="success", width = 150)
check_button.on_click(check_function)
# show the GUI
# widget boxes
wb1 = widgetbox(check_text,w_button,check_button,
                performance, height = height)
#wb2 = widgetbox(par_label2,w_d2,w_s1,w_s2,w_dt,
#                w_area,calibrate_button, height = height)
run_model_wb2 =row(upload_button, width=300)# widgetbox()#,w_area, height = height) 
#show(run_model_wb2 )
# make a grid
layout_runmodel= layout ( [[Main_title1],
                           [run_model_wb2] ,[precipitation_input_plot],
#                           [prec_filename],
                           [precipitation_slider,precipitatin_date],
                           [wb1, plot_qsim],#, wb2
                           [inputdata_title ],
                           [plot_sim, plot_evap],
                           ] )
    
run_model = Panel(child=layout_runmodel, title="Run the model")
''' ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================'''
#%% Input data

# show the GUI
input_data_Description = Div(text=" <h3> This section is for generating distributed data"
                             "from simple statistical methods like inverse distance weighting"
                             " method (IDWM) and inverse inverse squared distance weighting method (ISDWM) "
                             "<h3>",width = 1100)
#tbl_bw_label = Div(text="<h3> Table of Simulation Results <h3>"
#                   "<br>"
#                   "Note : At the time of this project, bokeh tables was unstable."
#                   "You may need to click and scroll in the table before data appears."
#                   ,width = 1200)

row1 = widgetbox(input_data_Description  )#,w_hn,w_dx,w_C,w_b,w_I,
#                  w_Nx,w_Q,w_button_bw, B_wd, B_export, w_files_bw)
layout_inputdata = layout([ [row1]#,p],
#                    [tbl_bw_label],
#                    [data_table_bw] 
                    ])
input_data = Panel(child=layout_inputdata, title="Input Data")

'''
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
'''
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

catchment_img= Div(text = "<img src='HBV_distributed/static/images/cover.jpg' "
                "style=width:500px;height:500px;vertical-align: middle >" )

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
#%%trial
#sp_quz_4km=np.load(data_file + "q_uz_c_4km.npy")
#
#s=dt.datetime(2012,6,14,19,0,0)
#e=dt.datetime(2013,12,23,0,0,0)
#
#dates=pd.date_range(s,e,freq="1H")
##n_years=e.year-s.year
#N=np.shape(sp_quz_4km)[2]
#
#
#
## flip the array as the column datasource already flip it 
#source = ColumnDataSource(data=dict(image=[np.flipud(sp_quz_4km[:, :, 0])]))
## color bar
#color_mapper=LogColorMapper(palette="Blues8", low=np.nanmin(sp_quz_4km),
#                            high=np.nanmax(sp_quz_4km), nan_color="white")
#color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
#                      border_line_color=None, location=(0,0)) #label_standoff=12
## figure
#TOOLS="pan, box_zoom, reset, save, crosshair"
#precipitation_input = figure( x_range=(0, np.shape(sp_quz_4km)[1]), y_range=(0, np.shape(sp_quz_4km)[0]),
#            toolbar_location="right",tools=TOOLS, plot_height=500, plot_width=500)#
#precipitation_input.image(image='image', x=0, y=0, dw=np.shape(sp_quz_4km)[1], dh=np.shape(sp_quz_4km)[0],
#         source=source,color_mapper=color_mapper)
#precipitation_input.add_layout(color_bar, 'right')
#
#
## TODO change the tag of the slider to day, hour, month and year 
#precipitation_slider= Slider(title="precipitation index ", start=0, end=(N-1), value=0, step=1, )#title="Frame"
#precipitatin_date = TextInput(value = str(dates[precipitation_slider.value]), title = 'date:',
#                              width=80)
##
#def update_precipitation(attr, old, new):
#    source.data = dict(image=[np.flipud(sp_quz_4km[:, :, precipitation_slider.value])])
#    precipitatin_date.value=str(dates[precipitation_slider.value])
#
#precipitation_slider.on_change('value', update_precipitation)
#
##show(p1)
#grid_trial=gridplot([[precipitation_input],[precipitation_slider,precipitatin_date]]) #[slider1]
##show(column(p1, slider1))
#%%
#year_slider = Slider(start=s.year, end=e.year, value=s.year, step=1, title="Frame")
#Month_slider= Slider(start=1, end=12, value=1, step=1, title="Frame")
#daily_slider= Slider(start=1, end=24, value=1, step=1, title="Frame")
#hourly_slider= Slider(start=0, end=(N-1), value=0, step=1, title="Frame")

#
#slider = Slider(start=1, end=360, value=250, step=1)
#
## Original dataset
#x = np.arange(0,len(rotateimg),1)
#y = [img[slider.value][i] for i in  xrange(len(x))]
#
#print(len(x))
#print(len(y))
#
## create a new plot with a title and axis labels
#p = figure(title="Brillo en una línea de la imagen", x_axis_label='x', y_axis_label='Brillo',tools=TOOLS)
#figline = p.line(x, y, line_width=2)
#
#def update_line(attr, old, new):
#    x = np.arange(0,len(rotateimg),1)
#    new_y = [img[slider.value][i] for i in  xrange(len(x))]
#
#    figline.data_source.data['y'] = new_y
#
#
#slider.on_change('value', update_line)
#layout = row(
#    widgetbox(slider),
#    p
#)
#
##output_file("slider.html", title="slider.py example")
#
#show(layout)
#%%
#x = [x*0.005 for x in range(0, 200)]
#y = x
#
#source = ColumnDataSource(data=dict(x=x, y=y))
#
#plot = figure(plot_width=400, plot_height=400)
#plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
#
#def callback(source=source, window=None):
#    data = source.data
#    # It passes the current plot_object as cb_obj implicitly.
#    f = cb_obj.value   # NOQA f is the power 
#    x, y = data['x'], data['y']
#    for i in range(len(x)):
#        y[i] = window.Math.pow(x[i], f)
#    source.change.emit();
#
#slider = Slider(start=0.1, end=4, value=1, step=.1, title="power",
#                callback=CustomJS.from_py_func(callback))
#
#interactive_plot= column(slider, plot)
#
#show(interactive_plot)

#ss=widgetbox( ,plot) 
#
#division=Div(text="<h3> ertywerty<h3>")
#division2=Div(text="<h3> ertywerty<h3>")
#
#def my_text_input_handler(attr, old, new):
#    division.text=old
#    division.text=new
#    print("Previous label: " + old)
#    print("Updated label: " + new)
#    
#
#text_input = TextInput(value="default", title="Label:")
#text_input.on_change("value", my_text_input_handler)
#show(column(text_input,division,division2))
#%%
#from os.path import dirname, join

#import datetime
#df = pd.read_csv('test.csv')
#df['dat'] = pd.to_datetime(df['date'])
#source1 = ColumnDataSource(data=dict())
##
##def update():
##    current = df[(df['dat'] >=  pd.to_datetime(sliderr.value[0])) & (df['dat'] <=  pd.to_datetime(sliderr.value[1]))]
##    source1.data = {
##        'opens'             : current.open,
##        'dates'           : current.open,
##      }
##sliderr = DateRangeSlider(title="Date Range: ", start=date(2010, 1, 1), end=date.today(), value=(date(2017, 9, 7),date.today()), step=1)
#
#sliderr = DateRangeSlider(title="Date Range: ", start=date(2010, 1, 1), end=date.today(), value=(date(2017, 9, 7),date(2017, 9, 7)), step=1)
#sliderr.on_change('value', lambda attr, old, new: update())
##
#columns = [
#   TableColumn(field="dates", title="Date" ,formatter=DateFormatter()),
#    TableColumn(field="opens", title="open"),]
##
#data_table = DataTable(source=source1, columns=columns, width=800)
##
#controls = widgetbox(sliderr)
#table = widgetbox(data_table)
#show(column(controls ,table ))

#grid_trial=gridplot([[controls ],[data_table]])
#%%
#trial=Panel(child = grid_trial, title = "trial")
#%% creating tabs
tabs = Tabs(tabs=[ hometab,input_data, calibrate_model, run_model,  ])#trial,
#tabs = Tabs(tabs=[ trial2  ])
curdoc().add_root(tabs)


