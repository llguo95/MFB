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
noise_type = 'b'
cost_ratio = 100

print()
print('dim = ', dim)
# print('noise_type = ', noise_type)
print('cost_ratio = ', cost_ratio)

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

model_type = ['sogpr', 'stmf']
lf = [.9]
n_reg = [5 ** dim]
n_reg_lf = [cost_ratio * 5 ** (dim - 1)]
scramble = 1
noise_fix = 0
budget = 3 * 5 ** dim
post_processing = 0
acq_type = 'EI'
iter_thresh = 25 * 3 ** (dim - 1)
dev = 0
# opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type
DoE_no = 10

start = time.time()

# bo_main(problem=problem, model_type=model_type, lf=lf, n_reg_init=n_reg, scramble=scramble, noise_fix=noise_fix,
#         n_reg_lf_init=n_reg_lf, max_budget=budget, post_processing=post_processing, acq_type=acq_type,
#         iter_thresh=iter_thresh, dev=dev, opt_problem_name=opt_problem_name)

for noise_type in ['b', 'n']:

    for acq_type in ['EI', 'UCB']:

        opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type

        for problem_el in problem:

            for model_type_el in model_type:

                for lf_el in lf:

                    for n_DoE in range(DoE_no):

                        start_unit = time.time()

                        bo_main_unit(problem_el=problem_el, model_type_el=model_type_el, lf_el=lf_el,
                                     n_reg_init_el=n_reg[0], n_reg_lf_init_el=n_reg_lf[0], scramble=scramble,
                                     noise_fix=noise_fix, noise_type=noise_type, max_budget=budget, acq_type=acq_type,
                                     iter_thresh=iter_thresh, dev=dev, opt_problem_name=opt_problem_name, n_DoE=n_DoE)

                        end_unit = time.time()
                        print('Unit run time', end_unit - start_unit)

stop = time.time()
print('Total run time', stop - start)

plt.show()