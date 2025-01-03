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

1.5.0 (2023-01-10)
------------------
* hydraulic model can read chunked big zip file
* fix CI
* fix missing module (saint venant script and module)

1.6.0 (2023-02-03)
------------------
* all attributes follows snake case naming convention
* refactor all modules with pre-commit
* add smoothDikeLevel, getReach and updateReach
* bump up denpendencies versions
* move un-necessary functions to serapeum-utils


2.0.0 (2025-01-**)
------------------

Dev
---
- Add conda/pypi workfkow
- Add github release workfkow
- Add pypi release workfkow
- remove coverall
- remove flake8 separate config file
- move the main packge files inside src directory
- move the hydrodymanic model to separate repo (serapis)
- move the plot module to the cleopatra package
- replace the setup.py by pyproject.toml