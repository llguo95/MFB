import time
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
# from main import reg_main
from main_new import reg_main
import pybenchfunction
from objective_formatter import botorch_TestFunction, AugmentedTestFunction

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]

dim = 1
noise_type = 'b'
post_processing = 1

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim), negate=False, # Minimization
            ), noise_type=noise_type,
        ).to(**tkwargs)
    )
    for f in fs
][2:3]

print([(i, f.name) for (i, f) in enumerate(fs)])

model_type = ['sogpr']
lf = [.5]
n_reg = [5, 10, 15, 20]
n_reg_lf = [1, 1, 1, 1]
scramble = 0
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
print('total time', stop - start)

plt.show()
