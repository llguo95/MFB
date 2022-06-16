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
cost_ratio = 25

print()
print('dim = ', dim)
print('noise_type = ', noise_type)
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
][:1]

model_type = ['sogpr', 'stmf']
lf = [.75]
n_reg = [5 ** dim]
n_reg_lf = [cost_ratio * 5 ** (dim - 1)]
scramble = 0
noise_fix = 0
budget = 3 * 5 ** dim
post_processing = 1
acq_type = 'EI'
iter_thresh = 50
dev = 1

trial = 0
start = time.time()

while trial < 1:
    bo_main(problem=problem, model_type=model_type, lf=lf, n_reg_init=n_reg, scramble=scramble, noise_fix=noise_fix,
            n_reg_lf_init=n_reg_lf, max_budget=budget, post_processing=post_processing, acq_type=acq_type,
            iter_thresh=iter_thresh, dev=dev)
    trial += 1
stop = time.time()
print('Total run time', stop - start)

plt.show()