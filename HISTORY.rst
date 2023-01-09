=======
History
=======

1.3.2 (2022-12-26)
------------------

* remove parameters from the package and retrieve them with the parameter package.


1.3.3 (2022-12-27)
------------------

* use joblib to parallelize reading laterals in hydraulic model


1.3.4 (2022-12-27)
------------------

* merge two function readLaterals and readRRMProgression, rename RRMProgression to routedRRM

1.3.5 (2022-12-27)
------------------

* fix pypi package names in the requirements.txt file for all internal packages
* fix python version number
* tests are all passing

1.4.0 (2022-12-27)
------------------

* remove fiona and the reading file exception using fion
* unify reading results of rainfall-runoff model in the readRRMResults, ReadLaterals, ReadUSHydrographs
* refactor code and change methods to camelcase
* add hydrodynamic model 1d config file read function
* simplify functions with too many parameters using decorator
* add automatic pypi build and publish github actions
