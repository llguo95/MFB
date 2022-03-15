import time
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
from main import reg_main
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
# print(len(fs))

# fs = [function.AlpineN2, function.Ridge, function.Schwefel, function.Ackley]
# fs = [function.AlpineN2]

dim = 1
noise_type = 'b'

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

model_type = ['sogpr']
lf = [.5]
n_reg = [5]
n_reg_lf = [15]
scramble = True
noise_fix = True

# model_type = ['mtask']
# lf = [.1, .5, .9]
# n_reg = [5]
# n_reg_lf = [15]
# scramble = True
# noise_fix = True

# model_type = ['mtask']
# lf = [.5]
# n_reg = [5]
# n_reg_lf = [30]
# scramble = True
# noise_fix = True

start = time.time()
data, metadata = reg_main(
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
print(stop - start)

# metadata['dim'] = dim
#
# folder_path = 'data2/'
# file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime())
#
# open_file = open(folder_path + file_name + '.pkl', 'wb')
# pickle.dump(data, open_file)
# open_file.close()
#
# open_file = open(folder_path + file_name + '_metadata.pkl', 'wb')
# pickle.dump(metadata, open_file)
# open_file.close()
#
# with open(folder_path + file_name + '_metadata.txt', 'w') as data:
#     data.write(str(metadata))

plt.show()
