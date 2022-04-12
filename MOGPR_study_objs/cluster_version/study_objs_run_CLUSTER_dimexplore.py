import sys

custom_lib_rel_path = '../'

sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')
import pybenchfunction

sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
import gpytorch

sys.path.insert(0, custom_lib_rel_path + 'GPy')
import GPy

sys.path.insert(0, custom_lib_rel_path + 'MFB')
from main import reg_main, bo_main

import pickle
import time
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
# from main import reg_main
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

dims = list(range(1, 11))
noise_type = 'b'
exp_type = 'm'

print()
print('noise_type = ', noise_type)

data, metadata = {}, {}
for dim in dims:
    print('dim = ', dim)

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
        lf = [.9]
        n_reg = [5 ** dim]
        n_reg_lf = [2 * (2 * k + 1) * 5 ** dim for k in range(15)]
        scramble = True
        noise_fix = 0

    start = time.time()
    data[dim], metadata[dim] = reg_main(
        problem=problem,
        model_type=model_type,
        lf=lf,
        n_reg=n_reg,
        n_reg_lf=n_reg_lf,
        scramble=scramble,
        noise_fix=noise_fix,
        noise_type=noise_type,
    )

stop = time.time()
print()
print('Run time:', stop - start)

folder_path = 'data/'
file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime())
suffix = '_' + noise_type + '_nf_' + str(noise_fix) + '_' + exp_type

open_file = open(folder_path + file_name + suffix + '.pkl', 'wb')
pickle.dump(data, open_file)
open_file.close()

open_file = open(folder_path + file_name + suffix + '_metadata.pkl', 'wb')
pickle.dump(metadata, open_file)
open_file.close()

with open(folder_path + file_name + suffix + '_metadata.txt', 'w') as data:
    data.write(str(metadata))

plt.show()
