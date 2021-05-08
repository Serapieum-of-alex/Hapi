*****
Sensetivity Analysis (OAT)
*****
If you want to check the sensitivity of the HBV hydrological model performance to predict stream flow to each parameter, the One-At-a Time sensitivity analysis is agreat meathod that helps in this area 
OAT fixes the value of all parameters and change the value of one parameter within boundaries each time to check the result of the given function based on different values of one of the inputs

First of all to run the HBV lumped model which we need to test its 
performance (RMSE) based on a defined range for each parameter 
``
import numpy as np
import pandas as pd

import Hapi.hbv_bergestrom92 as HBVLumped
import Hapi.run as RUN
from Hapi.routing import TriangularRouting
``
load all data needed to run the model as mentioned in [Lumped model](Lumped_HBV.md)
``
### meteorological data
path= comp + "/Coello/Hapi/Data/00inputs/Lumped/"
data=pd.read_csv(path+"meteo_data.txt",header=0 ,delimiter=',',
                   engine='python',index_col=0)
data=data.values

### Basic_inputs
ConceptualModel = HBVLumped
# p2 = [temporal resolution, catchment area]
p2=[24, 1530]
init_st=[0,10,10,10,0]
# no snow subroutine
snow = 0

### parameters
# parameters= pd.read_csv("results\parameters\lumped\oldparameters.txt", index_col = 0, header = None)
parameters = pd.read_csv(Parameterpath + "\scenario1.txt", index_col = 0, header = None)
parameters.rename(columns={1:'value'}, inplace=True)
parameters.drop('maxbas', axis=0, inplace=True)


UB = pd.read_csv(Parameterpath + "/UB-Extracted.txt", index_col = 0, header = None)
UB = UB[1].tolist()

LB = pd.read_csv(Parameterpath  + "/LB-Extracted.txt", index_col = 0, header = None)
LB = LB[1].tolist()


# observed flow
Qobs =np.loadtxt(path+"Qout_c.txt")
### Routing
Routing=1
RoutingFn=TriangularRouting

``
First the SensitivityAnalysis method takes 4 arguments :

    1-parameters:previous obtained parameters
    2-LB: upper bound
    3-UB: lower bound
    4-wrapper: defined function contains the function you want to run with different parameters and the metric function you want to assess the first function based on it.

Wrapper function definition
########

define the function to the OAT sesitivity wrapper and put the parameters argument
at the first position, and then list all the other arguments required for your function

the following defined function contains two inner function that calculates discharge for lumped HBV model and calculates the RMSE of the calculated discharge.

the first function "RUN.RunLumped" takes some arguments we need to pass it through the SensitivityAnalysis method 
[ConceptualModel,data,p2,init_st,snow,Routing, RoutingFn] with the same order in the defined function "wrapper"

the second function is RMSE takes the calculated discharge from the first function and measured discharge array

to define the argument of the "wrapper" function
1- the random parameters valiable i=of the first function should be the first argument "wrapper(Randpar)"
2- the first function arguments with the same order (except that the parameter argument is taken out and placed at the first potition step-1)
3- list the argument of the second function with the same order that the second function takes them

SensitivityAnalysis method returns a dictionary with the name of the parameters
as keys,
Each parameter has a disctionary with two keys 0: list of parameters woth relative values
1: list of parameter values

``
import matplotlib.pyplot as plt
import Hapi.performancecriteria as PC
import Hapi.statisticaltools as ST
``