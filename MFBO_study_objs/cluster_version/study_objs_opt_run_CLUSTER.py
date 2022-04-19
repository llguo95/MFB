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

import time
import pickle
import torch

from MFproblem import MFProblem
from main import bo_main
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

dim = 1
noise_type = 'b'
cost_ratio = 25

print()
print('dim = ', dim)
print('noise_type = ', noise_type)

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim),
            ), noise_type=noise_type, negate=True, # Minimization
        ).to(**tkwargs),
        cost_ratio=cost_ratio
    )
    for f in fs
][:1]

model_type = ['sogpr', 'cokg', 'stmf']
lf = [0.5, 0.9]
n_reg = [5 ** dim]
n_reg_lf = [2 * 5 ** dim]
scramble = False
noise_fix = 0
budget = 5 ** (dim + 1)

data_agg = []

start = time.time()
_ = 0
while _ < 1:
    data, metadata = bo_main(
        problem=problem,
        model_type=model_type,
        lf=lf,
        n_reg_init=n_reg,
        n_reg_lf_init=n_reg_lf,
        scramble=scramble,
        noise_fix=noise_fix,
        budget=budget,
    )
    data_agg.append(data)
    _ += 1
stop = time.time()
print()
print('Run time:', stop - start)

folder_path = 'data/'
file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime())
suffix = '_' + str(dim) + 'd_' + noise_type + '_nf_' + str(noise_fix) + '_cr_' + str(cost_ratio)

open_file = open(folder_path + file_name + suffix + '.pkl', 'wb')
pickle.dump(data_agg, open_file)
open_file.close()

open_file = open(folder_path + file_name + suffix + '_metadata.pkl', 'wb')
pickle.dump(metadata, open_file)
open_file.close()

with open(folder_path + file_name + suffix + '_metadata.txt', 'w') as data:
    data.write(str(metadata))