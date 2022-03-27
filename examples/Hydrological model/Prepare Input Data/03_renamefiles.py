from Hapi.inputs import Inputs as IN

"""
Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""

Path = "data/PrepareMeteodata/meteodata_prepared/temp-rename-example"

IN.RenameFiles(Path, fmt="%Y.%m.%d")
