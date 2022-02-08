PRELIMINARIES
1. Create an environment (e.g. conda) with the following off-the-shelf packages installed (e.g. with pip)
- numpy
- pandas
- torch
- matplotlib
- scipy
- gpytorch
- botorch
- emukit
- git
- sklearn

2. With git, clone the following packages without setup:
- pybenchfunction, from https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective.git
- GPy, from https://github.com/taylanot/GPy.git

3. Add these packages to your project structure.

GENERATING RESULTS
1. Run study_run.py in the teststudy folder. The parameters you might be interested in changing are:
- dim, the dimension of the objective
- LF, the LF parameter to augment the objective
- cost_ratio, the number of LF samples that attain the same amount of cost as 1 (one) HF sample
- noise_type, aleatoric uncertainty type which gets augmented into the LF objective; 'b' for only bias, 'n' for only noise, 'bn' for bias and noise
- model_type, list of model types; 'sogpr' for single-fidelity GPR, 'cokg' for cokgj, 'stmf' for multitask GPR
- n_reg and n_reg_lf, lists of initial DoE sizes
- scramble, boolean indicating doing a single run (False) or collecting statistics for 10 runs with different DoE locations (True)
- noise_fix, boolean indicating a noiseless (True) or noisy (False) HF objective
- budget, number to indicate max. units of cost expended during the optimization process

2. Three files with the same timestamp will be created. In study_processing.py, the base name without extension can be appended to the files_name list.

3. By running study_processing.py, the data dictionary can be taken apart to do further processing.