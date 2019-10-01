# -*- coding: utf-8 -*-
"""
Created on Tue May 08 19:53:25 2018

@author: Mostafa
"""
#%links

#%library
import numpy as np

from bokeh.io import curdoc, show
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput #,DateRangeSlider
from bokeh.plotting import figure
# functions


#%%


# Set up data
N = 200
x = np.linspace(0, 4*np.pi, N)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))


# Set up plot
plot = figure(plot_height=400, plot_width=400, title="my sine wave",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value

    # Generate the new curve
    x = np.linspace(0, 4*np.pi, N)
    y = a*np.sin(k*x + w) + b

    source.data = dict(x=x, y=y)

for w in [offset, amplitude, phase, freq]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(text, offset, amplitude, phase, freq)

#curdoc().add_root(row(inputs, plot, width=800))
#curdoc().title = "Sliders"
show(row(inputs, plot, width=800))
#%%
from os.path import dirname, join
import pandas as pd
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource#, CustomJS
from bokeh.models.widgets import DateRangeSlider,DatePicker,DateFormatter, DataTable, TableColumn, NumberFormatter
from bokeh.io import curdoc
from datetime import datetime,date
import datetime

df = pd.read_csv('test.csv')
df['dat'] = pd.to_datetime(df['date'])
source = ColumnDataSource(data=dict())

def update():
    current = df[(df['dat'] >=  pd.to_datetime(slider.value[0])) & (df['dat'] <=  pd.to_datetime(slider.value[1]))]
    source.data = {
        'opens'             : current.open,
        'dates'           : current.date,
      }

slider = DateRangeSlider(title="Date Range: ", start=date(2010, 1, 1), end=date.today(), value=(date(2017, 9, 7),date.today()), step=1)
slider.on_change('value', lambda attr, old, new: update())
    



columns = [
   TableColumn(field="dates", title="Date" ,formatter=DateFormatter()),
    TableColumn(field="opens", title="open"),]


data_table = DataTable(source=source, columns=columns, width=800)

controls = widgetbox(slider)
table = widgetbox(data_table)

curdoc().add_root(row(controls, table))
show(row(controls, table))
#update()
#%%
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