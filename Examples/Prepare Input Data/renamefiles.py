
from Hapi.inputs import Inputs as IN

Comp = "F:/Users/mofarrag/"
# Path = Comp + "/Coello/Hapi/Data/00inputs/meteodata/4000/calib/prec-CPC-NOAA"
Path = Comp + "/Coello/Hapi/Data/00inputs/meteodata/4000/valid/prec-CPC-NOAA"

IN.RenameFiles(Path, fmt = '%Y.%m.%d')

