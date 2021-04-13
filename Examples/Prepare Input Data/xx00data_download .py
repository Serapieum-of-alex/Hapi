import watools as wa

longs=[-75.630736,-74.837992]
lats=[4.172453,4.702915 ]

Stdate='2009-01-01'
Edate='2013-12-31'
directory = "C:/Users/Mostafa/Desktop/My Files/WA/colombia"
#%% precipitation
# chrips
#prec=wa.Collect.CHIRPS
#prec.daily(Dir="C:/Users/Mostafa/Desktop/My Files/WA/colombia",
#           latlim=lats , lonlim=longs,
#           Startdate='2009-03-20', Enddate=Edate,cores=True) #'2009-01-01' '2013-12-31'
#%% Temperature
ecmwf=wa.Collect.ECMWF
#ecmwf.daily(Dir="C:/Users/Mostafa/Desktop/My Files/WA/colombia",Vars=['T'],
#           latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
#           Startdate='2009-01-01', Enddate='2013-12-31')
#%% Evapotranspiration
ecmwf.daily(Dir="C:/Users/Mostafa/Desktop/My Files/WA/colombia",Vars=['E'],
           latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
           Startdate='2009-01-01', Enddate='2009-01-02')#'2013-12-31'
#%% Soil Type
ecmwf.daily(Dir=directory,Vars=['SLT'],
           latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
           Startdate='2009-01-01', Enddate='2013-12-31')
#%% evapotranspiration
ETref=wa.Products.ETref
Ett=ETref.daily(Dir="C:/Users/Mostafa/Desktop/My Files/WA/colombia",
                latlim=[4.190755,4.643963] , lonlim=[-75.649243,-74.727286],
                Startdate='2009-01-01', Enddate='2013-12-31')
