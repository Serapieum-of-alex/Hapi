"""
main page
"""
#import os
#os.chdir("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model")
import sys
sys.path.append("HBV_distributed/function")
#%% Library
import numpy as np
import pandas as pd
import time
import datetime as dt
#import gdal
from math import pi

from bokeh.layouts import widgetbox, gridplot, column
from bokeh.models.widgets import Slider, Button, RadioGroup, TextInput, Div, Tabs 
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Panel
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import curdoc, show
from bokeh.models import LinearAxis, Range1d

# functions
import DHBV_functions
#import sugawara as sug
#import fsf_model
#import bwc_collis as BW
#%% Read the output.asc file
skip_rows = 16  # Number of rows to skip in the output file
data_file = 'myapp/static/data/output.asc' # Name of the output file
data_file1= 'HBV_distributed/static/input_data/' # Name of the output file
#
s=dt.datetime(2012,06,14,19,00,00)
e=dt.datetime(2013,12,23,00,00,00)
#
#index=pd.date_range(s,e,freq="1H")
#lake_data=pd.DataFrame(index=index)
#lake_data['et']=np.loadtxt(data_file1+"lake/" + "et.txt")
#lake_data['tm']=np.loadtxt(data_file1+"lake/" + "avgtemp.txt")
#lake_data['plake']=np.loadtxt(data_file1+"lake/" + "plake.txt")
#lake_data['Q']=np.loadtxt(data_file1+"lake/" + "Q.txt")
#lake_data['t']=np.loadtxt(data_file1+"lake/" + "temp.txt")
#lakecell=[2,1] # 4km
#lakecell=[4,2] # 2km
#lakecell=[10,4] # 1km
#lakecell=[19,10] # 500m

#sp_prec_c=np.load(data_file1 +'sp_prec_c.npy')
#sp_et_c=np.load(data_file1 +'sp_et_c.npy')
#sp_temp_c=np.load(data_file1 +'sp_temp_c.npy')
#
#flow_acc_table=DHBV_functions.load_obj(data_file1 +"flow_acc_table")
#flow_acc=np.load(data_file1 +'flow_acc.npy')

#DEM = gdal.Open(data_file+"/DEM/"+"dem_4km.tif")
#elev, no_val=DHBV_functions.get_raster_data(DEM)
#
#elev[elev==no_val]=np.nan
#no_cells=np.size(elev[:,:])-np.count_nonzero(np.isnan(elev[:,:]))


# Read data from the output file
data = pd.read_csv(data_file,
                   skiprows=skip_rows,
                   skipinitialspace=True,
                   index_col='Time')
# Create vector with time stamps
time_index = pd.date_range('1994 12 07 20:00', periods=len(data), freq='H')
data.set_index(time_index, inplace=True)

# Intial Parameters 
pars = [0.5, 0.2, 0.01, 0.1, 10.0, 20.0, 1, 1]
extra_pars = [1, 147.0] # time factor and area (extra parameters)

# Define the precipitation data to give to the model
prec = np.array(data['Rainfall']) + np.array(data['Snowfall'])
evap = np.array(data['ActualET'])
q_rec = np.array(data['Qrec'])
snow = np.array(data['Snowfall'])
#%% Setup model (function)
q_sim = []  
#set up data source
# all input data
ds_rec = ColumnDataSource(dict(q_rec = q_rec,  
                           ds_time = time_index,
                           evap = evap,
                           prec = prec,
                           snow = snow))
# calculated discharge 
ds_sim = ColumnDataSource(dict( q_sim = q_sim , ds_time = time_index))

# Create Data Table
columns_sug = [
        TableColumn(field='ds_time', title="Date", 
                    formatter=DateFormatter(format = 'ddMyy')),
        TableColumn(field="prec", title="Precipitation"),
        TableColumn(field="snow", title="Snowfall"),
        TableColumn(field="evap", title="Actual ET"),
        TableColumn(field="q_rec", title="Recorded Discharge"),] 

data_table = DataTable(source=ds_rec, 
                       columns=columns_sug, 
                       width=2*630,height=340)

#%% plotting in sugawara tab
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

#%% make the widgets
# text input
w_k1 = TextInput(value = '0.5', title = 'Upper tank upper Q coefficient')
w_k2 = TextInput(value = '0.2', title = 'Upper tank lower Q coefficient')
w_k3 = TextInput(value = '0.01', title = 'Percolation to lower tank coefficient')
w_k4 = TextInput(value = '0.1', title = 'Lower tank Q coefficient')

w_d1 = TextInput(value = '10.0', title = 'Upper tank upper Q position')
w_d2 = TextInput(value = '20.0', title = 'Upper tank lower Q position')
w_s1 = TextInput(value = '1.0', title = 'Level of the top tank [mm]')
w_s2 = TextInput(value = '1.0', title = 'Level of the bottom tank [mm]')

w_dt = TextInput(value = '1.0', title = 'Number of hours in the time step [s]')
w_area = TextInput(value = '147.0', title = 'Catchment area [km2]')

# buttons
w_button = Button(label = 'Run model', button_type = 'success' , width = 150)
calibrate_button = Button(label = 'Calibrate model', button_type = 'warning', width = 150)

# message
nse = Div(text=" ")
#%% define the update
def run_sugawara_model():
    nse.text = str("<h2>processing...<h2>")
    
    # read values of parameters 
    _k1 = float(w_k1.value)
    _k2 = float(w_k2.value)
    _k3 = float(w_k3.value)
    _k4 = float(w_k4.value)
    _d1 = float(w_d1.value)
    _d2 = float(w_d2.value)
    _s1 = float(w_s1.value)
    _s2 = float(w_s2.value)
    _dt = float(w_dt.value)
    _area = float(w_area.value)
    
    pars = [_k1, _k2, _k3, _k4, _d1, _d2, _s1, _s2]
    extra_pars = [_dt, _area]
    
    #run the model with the value of the interface
    q_sim, st_sim = sug.simulate(prec, evap, pars, extra_pars)  # Run the model
    
    #update data source
    ds_sim.data = (dict(q_sim = q_sim , ds_time = time_index))

    # Calculate model performance
    model_perf(q_sim, q_rec)    
 

def model_perf(q_sim, q_rec):
    q_sim.pop() # remove last element before NSE
    nse.text = str("<h2>calculating model performance..<h2>")
    perf = sug.NSE(q_sim, q_rec)
    nse.text = str("<h2>Model perfomance(NSE) is %s<h2>" %round(perf, 3))
    
    
def calibrate_sugawara_model():
    nse.text = str("<h2>calibrating...<h2>")
    
    x, fun = sug.calibrate(prec, evap, extra_pars, q_rec)
    
    # update text
    w_k1.value = str(x[0])
    w_k2.value = str(x[1])
    w_k3.value = str(x[2])
    w_k4.value = str(x[3])
    w_d1.value = str(x[4])
    w_d2.value = str(x[5])
    w_s1.value = str(x[6])
    w_s2.value = str(x[7])
    
    # update NSE
    nse.text = str("<h2>model calibrated, parameters updated, rerun model.<h2>")    
#%% assign buttons
w_button.on_click(run_sugawara_model)
calibrate_button.on_click(calibrate_sugawara_model)

div = Div(text="<h1 style=color:blue;>Sugawara Tank Model<h1>",
          width = 590, height=height)

par_label = Div(text=" <h3> Sugawara Model\n <h3>")
par_label2 = Div(text="<h3> Input Parameters\n <h3>")
model_label = Div(text="<h3>Model configuration and results<h3>")
file_label = Div(text="<h3>Input Data from file<h3>")
#%% show the GUI
# widget boxes
wb1 = widgetbox(par_label,w_k1,w_k2,w_k3,w_k4,w_d1,w_button, 
                nse, height = height)

wb2 = widgetbox(par_label2,w_d2,w_s1,w_s2,w_dt,
                w_area,calibrate_button, height = height)
#%% make a grid
grid = gridplot ( [[model_label ],
                   [wb1, wb2, plot_qsim] ,
                   [file_label ],
                   [plot_sim, plot_evap],
                   #[tbl_label ],
                   #[data_table ] 
                   ] )
tab2 = Panel(child=grid, title="SUGAWARA MODEL")

''' ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================
# ============================================================================================================================================================'''
#%% Back water curve 

# Setup model (function)
Model = BW.calcFixed

# Setup data
hn = 6.0
dx = 100.0
Q = 500
C = 50.0
b = 100.0
I = 0.001
Nx = 50


#==============================================================================
# pfile_name = 'myapp/static/data/input.txt'
# bw_pars = BW.readValues(pfile_name)
# 
# # assign each value to the key
# for k,v in bw_pars.items():     
#     exec("%s=%r" % (k,v))
# 
#==============================================================================
depth, hg, waterlevel, distance = Model(hn,dx,Q,C,b,I,Nx)

#set up data source
ds_bw = ColumnDataSource(dict(dist = distance,wl=waterlevel,z0 = hg,h = depth))

columns_bw = [TableColumn(field="dist", title="distance"),
              TableColumn(field="z0", title="bed level"),
              TableColumn(field="wl", title="water level"),
              TableColumn(field="h", title="water depth")
             ]

data_table_bw = DataTable(source=ds_bw, columns=columns_bw, 
                          width=1200, height=580)

#set up plot
p = figure(plot_width=width+300, plot_height=height+50,
           title = 'Back Water Curve' , x_range=(0, distance[-1]))

p.line(x = 'dist', y = 'wl', source = ds_bw, 
       alpha=0.5, color="navy", legend="Water level", line_width = 3)
p.line(x = 'dist', y = 'z0', source = ds_bw,
       alpha=0.5, color="black", legend="Bed level" , line_width = 3)
p.yaxis.axis_label = "Height (m)"
p.xaxis.axis_label = "Distance (m)"
p.legend.location = 'top_right'
p.legend.label_text_font_style = "italic"
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.xaxis.major_label_orientation = pi/4

#make the widgets
w_hn = TextInput(value = str(hn) , title = 'Initial depth h0')
w_dx = TextInput(value = str(dx), title = 'delta x dx')
w_C = TextInput(value = str(C), title = 'chezy coefficient C')
w_b = TextInput(value = str(b), title = 'channel width b')
w_I = TextInput(value = str(I), title = ' channel slope I')
w_Nx = TextInput(value = str(Nx), title = 'Number of iterations Nx')
w_Q = Slider(start=1, end=2000, value=Q, step=.1, title="Discharge") 
w_button_bw = Button(label = 'Run model', button_type = 'success', width = 150)
B_ehead = Div(text="<b>Export Results of Model</b>")
B_export = Button(label = 'Export Results', button_type = 'success', width = 150)
B_wd = TextInput(value = 'Backwater.csv', title = 'Enter file name for export:')
w_files_bw = Div(text = " ")

# define the update
def run_bwc_model():
    _hn = float(w_hn.value)
    _dx = float(w_dx.value)
    _C = float(w_C.value)
    _b = float(w_b.value)
    _I = float(w_I.value)
    _Nx = float(w_Nx.value)
    _Q = w_Q.value
    
    #run the model with the value of the interface
    depth,hg,waterlevel,distance = Model(_hn,_dx,_Q,_C,_b,_I,_Nx)
    
    R = np.zeros([len(depth),5])
    R[:,0] = range(len(depth))
    R[:,1] = distance
    R[:,2] = depth
    R[:,3] = hg
    R[:,4] = waterlevel
    
    #update the plot dimension
    p.x_range.end = _Nx*_dx
    p.y_range.end = 1.25*np.amax(np.array(waterlevel))

    #update data source
    ds_bw.data = dict(dist = distance, wl = waterlevel, z0 = hg, h = depth)
    
    
    def writefile():
        BackWaterOut = "myapp/results/%s" %str(B_wd.value)
        w_files_bw.text = str("writing files...")
        BW.createOutput(R, BackWaterOut)
        w_files_bw.text = str("files saved to myapp/results")
        
    B_export.on_click(writefile)
    
w_button_bw.on_click(run_bwc_model)


#%%show the GUI
par_bw_label = Div(text=" <h3> Parameters <h3>")
tbl_bw_label = Div(text="<h3> Table of Simulation Results <h3>"
                   "<br>"
                   "Note : At the time of this project, bokeh tables was unstable."
                   "You may need to click and scroll in the table before data appears."
                   ,width = 1200)
wb_bw = widgetbox(par_bw_label,w_hn,w_dx,w_C,w_b,w_I,
                  w_Nx,w_Q,w_button_bw, B_wd, B_export, w_files_bw)
grid_bw = gridplot([ [wb_bw,p],
                    [tbl_bw_label],
                    [data_table_bw] 
                    ])
tab1 = Panel(child=grid_bw, title="BACKWATER CURVE")

# =============================================================================

dx = 20
dt = 20
lgth = 1000
TimeMAX = 100

DepthIn = 'myapp/static/data/Depth.inp'
DischargeIn = 'myapp/static/data/Discharge.inp'

Ufile = 'myapp/static/data/ubc.txt'
Dfile = 'myapp/static/data/dbc.txt'


timenew = []
hini = []
qini = []
distance = []
WL = []
hg = []


dsu = ColumnDataSource(dict(h = hini, q = qini, time = timenew))
dsm = ColumnDataSource(dict(h = hini, q = qini, time = timenew))
dsd = ColumnDataSource(dict(h = hini, q = qini, time = timenew))

dswl = ColumnDataSource(dict(dist=distance,  wl = WL, hg = hg))
dsq = ColumnDataSource(dict(dist=distance,  q = qini))

# setup plot
ph = figure(x_range=Range1d(0, 1), y_range=Range1d(0, 1), width = 700, height = 280, title="Longitudinal Profile (Water Depth)")
ph.line(x = 'dist', y = 'wl', source = dswl, color = 'blue', line_width=2, legend = 'Water Level')
ph.line(x = 'dist', y = 'hg', source = dswl, color = 'grey', line_width=2, legend = 'Bed Level')
ph.legend.location = "top_right"
ph.xaxis.axis_label = "Distance (m)"
ph.yaxis.axis_label = "Water Level (m)"
#ph.xaxis.major_label_orientation = pi/4

pq = figure(x_range=Range1d(0, 1), y_range=Range1d(0, 1), width = 700, height = 280, title="Longitudinal Profile (Discharge)")
pq.line(x = 'dist', y = 'q', source = dsq, color = 'red', line_width=2)
pq.xaxis.axis_label = "Distance (m)"
pq.yaxis.axis_label = "Discharge (m3/s)"
#pq.xaxis.major_label_orientation = pi/4

thu = figure(x_range=Range1d(0, 1), y_range=Range1d(0, 1), width = 500, height = 400, title="Time Series Water depth")
thu.line(x = 'time', y = 'h', source = dsu, color = 'green', legend = 'upstream')
thu.line(x = 'time', y = 'h', source = dsm, color = 'red', legend = 'mid-channel')
thu.line(x = 'time', y = 'h', source = dsd, color = 'blue', legend = 'downstream')
thu.legend.location = "top_right"
thu.xaxis.axis_label = "Time (sec)"
thu.yaxis.axis_label = "Water depth (m)"
thu.xaxis.major_label_orientation = pi/4

thd = figure(x_range=Range1d(0, 1), y_range=Range1d(0, 1), width = 500, height = 400, title="Time Series Discharge")
thd.line(x = 'time', y = 'q', source = dsu, color = 'green', legend = 'upstream')
thd.line(x = 'time', y = 'q', source = dsm, color = 'red', legend = 'mid-channel')
thd.line(x = 'time', y = 'q', source = dsd, color = 'blue', legend = 'downstream')
thd.legend.location = "top_right"
thd.xaxis.axis_label = "Time (sec)"
thd.yaxis.axis_label = "Discharge (m3/s)"
thd.xaxis.major_label_orientation = pi/4

# make the widgets
Inhead = Div(text="<h3>Input for Free Surface Flow Model<h3>")
I_dx = TextInput(value = '500', title = 'Space Interval (m)')
I_dt = TextInput(value = '500', title = 'Time step (sec)')
I_TimeMAX = TextInput(value = '86400', title="Simulation Time (sec)")
I_NMAXIts = TextInput(value = '5', title="Maximum Iteration")
I_theta = TextInput(value = '0.55', title="Theta")
I_Psi = TextInput(value = '0.5', title="Psi")
I_Beta = TextInput(value = '1.0', title="Beta")
head2 = Div(text="<i><u>Physical Parameters:</u></i>")
I_b = TextInput(value = '100', title = 'Channel Width (m)')
I_lgth = TextInput(value = '10000', title = 'Channel Length (m)')
I_Ib = TextInput(value = '0.0001', title = 'Bed Slope')
I_C = TextInput(value = '50', title = 'Chezy Coefficient')

# defining boundary condition
I_ub = Div(text="<b>Upstreame Boundary Condition</b>")
uc_type = RadioGroup(labels=["Discharge", "Water depth"], active=0)
#uc_unit = Select(title="Time unit", value="hours", options=["days", "hours", "min", "sec"])
I_db = Div(text="<br><b>Downstream Boundary Condition</b>")
dc_type = RadioGroup(labels=["Discharge", "Water depth"], active=1)
#dc_unit = Select(title="Time unit", value="hours", options=["days", "hours", "min", "sec"])
I_initial = Div(text="<br><b>Initial Condition</b>")
I_qini = TextInput(value = '315', title = 'Initial Discharge (m3/s)')
I_hini= TextInput(value = '3.0', title = 'Initial Water Depth (m)')
blank2 = Div(text="<br>")

I_run = Button(label = 'Run Model', button_type = 'success', width = 150)
I_animate = Button(label = 'Start Animation', button_type = 'success', width = 150)

I_ehead = Div(text="<b>Export Water depth and Discharge</b>")
I_export = Button(label = 'Export Results', button_type = 'success', width = 150)
I_wd = TextInput(value = 'WaterDepth.txt', title = 'Enter fine name for depth:')
I_wq = TextInput(value = 'Discharge.txt', title = 'Enter fine name for discharge:')
w_files = Div(text = " ")

# define the model
def model_sim ():
    dx = int(I_dx.value)
    dt = int(I_dt.value)
    TimeMAX = int(I_TimeMAX.value)
    NMAXIts = int(I_NMAXIts.value)
    theta = float(I_theta.value)
    Psi = float(I_Psi.value)
    Beta = float(I_Beta.value)
    b = float(I_b.value)
    C = float(I_C.value)
    g = 9.81
    Ib = float(I_Ib.value)
    lgth = int(I_lgth.value)
    M = 1+int(lgth/dx)
    N = 1+int(TimeMAX/dt)
    
    if uc_type.active == 0:
        UC = 'Q'
    else:
        UC = 'h'	
    Ufile = 'myapp/static/data/ubc.txt'
    if dc_type.active == 0:
        DC = 'Q'
    else:
        DC = 'h'	
    Dfile = 'myapp/static/data/dbc.txt'

    

    ubc, dbc, timenew = fsf_model.readboundary (Ufile, Dfile, dt, TimeMAX)
    hini, qini, distance = fsf_model.readini (DepthIn, DischargeIn, dx, lgth)

    Q, h, hg, WL = fsf_model.fsfCalculation (dx, dt, TimeMAX, NMAXIts, theta, Psi, Beta, b, C, g, Ib, lgth, UC, DC, ubc, dbc, hini, qini)
    
     #update the plot dimension
    ph.x_range.end = pq.x_range.end = 1.02*lgth
    ph.y_range.end = thu.y_range.end =1.25*np.amax(np.array(WL))
    pq.y_range.end = thd.y_range.end = 1.25*np.amax(np.array(Q))    
    thu.x_range.end = thd.x_range.end = TimeMAX
    
    dsu.data=dict(h = h[:,0], q = Q[:,0], time = timenew)
    dsm.data=dict(h = h[:,int(lgth/(2*dx))], q = Q[:,int(lgth/(2*dx))], time = timenew)
    dsd.data=dict(h = h[:,-1], q = Q[:,-1], time = timenew)
    dswl.data=dict(dist=distance,  wl = WL[0], hg = hg[0])
    dsq.data=dict(dist=distance,  q = Q[0])

    def animation ():
        for i in range (len(h)):
            dswl.data=dict(dist=distance,  wl = WL[i], hg = hg[0])
            dsq.data=dict(dist=distance,  q = Q[i])
            time.sleep(0.2)
    
    # Show the GUI
    I_animate.on_click(animation)
    
    def writefile():
        DepthOut = "myapp/results/%s" %str(I_wd.value)
        DischargeOut = "myapp/results/%s" %str(I_wq.value)
        w_files.text = str("writing files...")
        
        with open(DischargeOut, 'w') as fname:
            # writing the heading
            fname.write('Computed Discharge (Q)\n')
            fname.write('Time ')
            for i in range (0, M):
                fname.write('Q_{:d} ' .format(i))
            fname.write('\n')
        
        with open(DepthOut, 'w') as fname:
            # writing the heading
            fname.write('Computed Water Depth (h)\n')
            fname.write('Time ')
            for i in range (0, M):
                fname.write('h_{:d} ' .format(i))
            fname.write('\n')
        
        for t in range (0, N): 
            with open(DischargeOut, 'a') as fname:
                fname.write('{:4d} ' .format(t))
                for z in range (0, M):
                    fname.write('{:4.1f} ' .format(Q[t][z]))
                fname.write('\n')
            with open(DepthOut, 'a') as fname:
                fname.write('{:4d} ' .format(t))
                for z in range (0, M):
                    fname.write('{:4.1f} ' .format(h[t][z]))
                fname.write('\n')
                
        w_files.text = str("files saved to myapp/results")
        
    # write output
    I_export.on_click(writefile)

I_run.on_click(model_sim)

wb_fsf = widgetbox(I_dx, I_dt, I_TimeMAX, I_NMAXIts, I_theta, I_Psi, I_Beta, I_initial, I_qini, I_hini )
wb2_fsf = widgetbox(I_b, I_lgth,I_Ib,I_C, I_ub, uc_type, I_db, dc_type, I_run, I_animate )
col2_fsf = column(ph, pq)
export_fsf = widgetbox(I_ehead, I_wd, I_wq, I_export, w_files)

 
grid_fsf = gridplot ( [ [Inhead],
	                   [wb_fsf, wb2_fsf, col2_fsf] ,
                       [thu, thd, export_fsf]
                   ] )

tab3 = Panel(child=grid_fsf, title="FREE SURFACE FLOW")
'''
# ==========================================================================================================================================================
==========================================================================================================================================================
==========================================================================================================================================================
'''
#%% Home Page
home_div = Div(text="<h1 style=color:DodgerBlue;"
               "font-size:50px;font-family:comic sans ms >"
               "Modelling Systems Development Project<h1>",
               width = 1100)

intro_div = Div(text="<h1>Introduction<h1>"
               "<h3>This project combines the contents and ideas of "
               "of module 5 of WSE/HI 2017-2019. It was developed using "
               "python programming language and bokeh for visual interaction."
               "Meet the developers"
               "<h3>",
               width = 1100)

img_allen = Div(text = "<img src=myapp/static/images/allen.jpg "
                "style=width:240px;height:240px;>" )

desc_allen = Div(text = "<h3> Name: Colis Allen <h3>"
                 "Background: Computer Science"
                 "<br>"
                 "Country: Guyana (South America)"
                 "<br>"
                 "Role: Sugawara Tank Model GUI and Project Compilation")

img_dianah = Div(text = "<img src=myapp/static/images/dianah.jpg "
                "style=width:240px;height:240px;>" )

desc_dianah = Div(text = "<h3> Name: Dianah Nasasira <h3>"
                 "Background: Civil Engineering"
                 "<br>"
                 "Country: Uganda"
                 "<br>"
                 "Role: BackWater Curve GUI and Report")

img_harsha = Div(text = "<img src=myapp/static/images/harsha.jpg "
                "style=width:240px;height:240px;>" )

desc_harsha = Div(text = "<h3> Name: Harsha Abeykoon <h3>"
                 "Background: Civil Engineering"
                 "<br>"
                 "Country: Sri Lanka"
                 "<br>"
                 "Role: Testing and Report")

img_tanvir = Div(text = "<img src=myapp/static/images/tanvir.jpg "
                "style=width:240px;height:240px;>" )

desc_tanvir = Div(text = "<h3> Name: Tanvir Ahmed <h3>"
                 "Background: Water Resources Engineer"
                 "<br>"
                 "Country: Bangladesh"
                 "<br>"
                 "Role: Free Surface Flow GUI and Design")


wb_home = widgetbox(home_div)

grid_home = gridplot( [ [wb_home],
                        [intro_div],
                        [img_allen , desc_allen, img_dianah, desc_dianah],
                        [img_harsha, desc_harsha, img_tanvir, desc_tanvir],
                      ]
                    )

hometab = Panel(child = grid_home, title = "HOME")
#%% creating tabs

tabs = Tabs(tabs=[ hometab, tab1, tab2, tab3])
curdoc().add_root(tabs)


