"""
Created on Wed Jul 18 23:55:20 2018

@author: Mostafa
"""


import datetime
#%library
import pickle

# import numpy as np

# functions
def save_obj(obj, saved_name):
    """
    ===============================================================
        save_obj(obj, saved_name )
    ===============================================================
    this function is used to save any python object to your hard desk

    Inputs:
    ----------
        1-obj:

        2-saved_name:
            ['String'] name of the object
    Outputs:
    ----------
        the object will be saved to the given path/current working directory
        with the given name
    Example:
        data={"key1":[1,2,3,5],"key2":[6,2,9,7]}
        save_obj(data,path+'/flow_acc_table')
    """
    with open(saved_name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(saved_name):
    """
    ===============================================================
        load_obj(saved_name)
    ===============================================================
    this function is used to save any python object to your hard desk

    Inputs:
    ----------
        1-saved_name:
            ['String'] name of the object
    Outputs:
    ----------
        the object will be loaded
    Example:
        load_obj(path+'/flow_acc_table')
    """
    with open(saved_name + ".pkl", "rb") as f:
        return pickle.load(f)


def dateformated(x):
    """
    ===========================================================
        dateformated(x)
    ===========================================================
    this function converts the the date read from a list to a datetime format

    input:
        x is a list of tuples of string date read from database

    output:
        list od dates as a datetime format  YYYY-MM-DD HH:MM:SS

    """
    x = [i[0] for i in x]
    #
    x1 = []
    for i in x:
        if len(i) == 19:
            x1.append(
                datetime.datetime(
                    int(i[:4]),
                    int(i[5:7]),
                    int(i[8:10]),
                    int(i[11:13]),
                    int(i[14:16]),
                    int(i[17:18]),
                )
            )
    #        elif len(i)==13:
    #            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]),int(0),int(0) ))
    #        else:
    #            x1.append(datetime.datetime(int(i[:4]),int(i[5:7]),int(i[8:10]),int(0),int(0),int(0) ))
    #    del i,x
    return x1


def printWaitBar(i, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    """
    This function will print a waitbar in the console

    Variables:

    i -- Iteration number
    total -- Total iterations
    fronttext -- Name in front of bar
    prefix -- Name after bar
    suffix -- Decimals of percentage
    length -- width of the waitbar
    fill -- bar fill
    """
    import os
    import sys

    # Adjust when it is a linux computer
    if os.name == "posix" and total == 0:
        total = 0.0001

    percent = ("{0:." + str(decimals) + "f}").format(100 * (i / float(total)))
    filled = int(length * i // total)
    bar = fill * filled + "-" * (length - filled)

    sys.stdout.write("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    if i == total:
        print()
