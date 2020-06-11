# Welcome to Hapi 
Distributed Hydrological Model





## Purpose

Discription

```
some code
```


## Features


* Available algorithms are (`HBV`).
* Nash-Sutcliff (`NSE`), log Nash-Sutcliff (`logNSE`), Root Mean Squared Error (`RMSE`), Mean Absolute Error (`MAE`).
  Kling-Gupta Efficiency (`KGE`).



## Installation

### Dependencies

* [NumPy](http://www.numpy.org/ "Numpy")
* [Scipy](http://www.scipy.org/ "Scipy")

Optional packages are:

* [Matplotlib](http://matplotlib.org/ "Matplotlib")
* [Pandas](http://pandas.pydata.org/ "Pandas")
* [mpi4py](http://mpi4py.scipy.org/ "mpi4py")

### Download

	pip install Hapi


## Project layout



*Above: Overview about functionality of the Hapi package*


	
	__init__.py             # Ensures that all needed files are loaded.
	
    algorithms/
        __init__.py   # Ensures the availability of all algorithms
	
	parallel/
		mpi.py        #Basic Parralel Computing features 

	examples/
		3dplot.py                   # Response surface plot of example files

