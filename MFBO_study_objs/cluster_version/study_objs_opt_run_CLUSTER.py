import sys

custom_lib_rel_path = '../'

sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')
import pybenchfunction

sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
import gpytorch

sys.path.insert(0, custom_lib_rel_path + 'GPy')
import GPy

sys.path.insert(0, custom_lib_rel_path + 'MFB')
# from main import reg_main, bo_main

import time
import pickle
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
from main_new import bo_main
import pybenchfunction
from pybenchfunction import function
from objective_formatter import botorch_TestFunction, AugmentedTestFunction

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
print()
print([(i, f(d=1).get_global_minimum(d=1)[1]) for (i, f) in enumerate(fs)])

dim = 1
noise_type = 'b'
cost_ratio = 15

print()
print('dim = ', dim)
print('noise_type = ', noise_type)

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim),
            ), noise_type=noise_type, negate=False, # Minimization
        ).to(**tkwargs),
        cost_ratio=cost_ratio
    )
    for f in fs
]

model_type = ['sogpr']
lf = [0.9]
n_reg = [5 ** dim]
n_reg_lf = [2 * 5 ** dim]
scramble = 1
noise_fix = 0
budget = 3 * 5 ** dim
post_processing = False

trial = 0
start = time.time()
while trial < 1:
    bo_main(problem=problem, model_type=model_type, lf=lf, n_reg_init=n_reg, scramble=scramble, noise_fix=noise_fix,
            n_reg_lf_init=n_reg_lf, max_budget=budget, post_processing=post_processing)
    trial += 1
stop = time.time()
print('Total run time', stop - start)

plt.show()