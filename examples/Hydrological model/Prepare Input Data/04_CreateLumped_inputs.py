"""Created on Fri Feb 26 06:07:08 2021.

@author: mofarrag
"""
"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""
import numpy as np

from Hapi.inputs import Inputs as IN

Path = "data/PrepareMeteodata/meteodata_prepared/temp-lumped-example"
SaveTo = "data/PrepareMeteodata/meteodata_prepared/lumped_temp.txt"

data = IN.CreateLumpedInputs(Path)
np.savetxt(SaveTo, data, fmt="%7.2f")
