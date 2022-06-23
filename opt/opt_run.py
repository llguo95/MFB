import sys

custom_lib_rel_path = '../../'

sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')
import pybenchfunction

sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
import gpytorch

sys.path.insert(0, custom_lib_rel_path + 'GPy')
import GPy

sys.path.insert(0, custom_lib_rel_path + 'MFB')

import time
import pickle
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
from main_new import bo_main, bo_main_unit
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

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim),
            ), negate=False,
        ).to(**tkwargs)
    )
    for f in fs
]

model_type = ['stmf']
lf = [.9]
noise_types = ['b']
acq_types = ['UCB']
cost_ratio = 10
n_reg = [5 ** dim]
n_reg_lf = [cost_ratio * 5 ** (dim - 1)]
scramble = 1
noise_fix = 0
budget = 5 * 3 ** (dim - 1)
iter_thresh = 50 * 3 ** (dim - 1)

print()
print('dim = ', dim)
print('cost_ratio = ', cost_ratio)

dev = 1
DoE_no = 10
exp_name = 'exp4'

start = time.time()

for noise_type in noise_types:

    for acq_type in acq_types:

        opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type

        for problem_el in problem:

            for model_type_el in model_type:
                if model_type_el == 'sogpr':
                    exp_name += '_sogpr'

                for lf_el in lf:

                    for n_DoE in range(DoE_no):

                        start_unit = time.time()

                        bo_main_unit(problem_el=problem_el, model_type_el=model_type_el, lf_el=lf_el, cost_ratio=cost_ratio,
                                     n_reg_init_el=n_reg[0], n_reg_lf_init_el=n_reg_lf[0], scramble=scramble,
                                     noise_fix=noise_fix, noise_type=noise_type, max_budget=budget, acq_type=acq_type,
                                     iter_thresh=iter_thresh, dev=dev, opt_problem_name=opt_problem_name, n_DoE=n_DoE, exp_name=exp_name)

                        end_unit = time.time()
                        print('Unit run time', end_unit - start_unit)

stop = time.time()
print('Total run time', stop - start)

# plt.show()