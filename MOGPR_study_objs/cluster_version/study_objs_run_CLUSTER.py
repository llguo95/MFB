import sys

custom_lib_rel_path = '../'

sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')
import pybenchfunction

sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
import gpytorch

sys.path.insert(0, custom_lib_rel_path + 'GPy')
import GPy

sys.path.insert(0, custom_lib_rel_path + 'MFB')
from main_new import reg_main, bo_main

import pickle
import time
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
import pybenchfunction
from objective_formatter import botorch_TestFunction, AugmentedTestFunction

print()
print('Imports succeeded!')

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]
print()
print([(i, f.name) for (i, f) in enumerate(fs)])

dim = 1
noise_type = 'b'
exp_type = 'm'
post_processing = 1

print()
print('dim = ', dim)
print('noise_type = ', noise_type)
print('post processing = ', post_processing)

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim), negate=False, # Minimization
            ), noise_type=noise_type,
        ).to(**tkwargs)
    )
    for f in fs
]

if exp_type == 's':

    model_type = ['sogpr']
    lf = [.5]
    n_reg = [6 * 5 ** (dim - 1)]
    n_reg_lf = [5 ** dim]
    scramble = True
    noise_fix = 0

elif exp_type == 'm':

    model_type = ['cokg', 'cokg_dms', 'mtask']
    lf = [.1, .5, .9]
    n_reg = [5 ** dim] * 4
    n_reg_lf = [(3 * k + 1) * 5 ** dim for k in range(4)]
    scramble = True
    noise_fix = 0

start = time.time()
reg_main(
    problem=problem,
    model_type=model_type,
    lf=lf,
    n_reg=n_reg,
    n_reg_lf=n_reg_lf,
    scramble=scramble,
    noise_fix=noise_fix,
    noise_type=noise_type,
    optimize=1 - post_processing,
)
stop = time.time()
print()
print('Run time:', stop - start)

plt.show()
