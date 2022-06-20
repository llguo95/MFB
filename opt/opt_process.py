# import sys
#
# custom_lib_rel_path = '../../'
#
# sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')
# import pybenchfunction
#
# sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
# import gpytorch
#
# sys.path.insert(0, custom_lib_rel_path + 'GPy')
# import GPy
#
# sys.path.insert(0, custom_lib_rel_path + 'MFB')

import time
import pickle

import pandas as pd
import torch
from matplotlib import pyplot as plt

from GPy.models.gradient_checker import np
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
cost_ratio = 25

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
budget = 5 * 3 ** dim
post_processing = 0
acq_type = 'EI'
iter_thresh = 25 * 3 ** (dim - 1)
dev = 0
# opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type
DoE_no = 10

start = time.time()

exp_dict = {
    'sogpr': 'exp2_sogpr',
    'stmf': 'exp2',
}

# bo_main(problem=problem, model_type=model_type, lf=lf, n_reg_init=n_reg, scramble=scramble, noise_fix=noise_fix,
#         n_reg_lf_init=n_reg_lf, max_budget=budget, post_processing=post_processing, acq_type=acq_type,
#         iter_thresh=iter_thresh, dev=dev, opt_problem_name=opt_problem_name)

for noise_type in ['b']:

    for acq_type in ['EI']:

        meds_model = []
        for model_type_el in model_type:
            print(model_type_el)

            if model_type_el == 'sogpr':
                opt_problem_name = str(dim) + '_b_' + acq_type
            else:
                opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type
            # print(opt_problem_name)

            meds = []
            for problem_i, problem_el in enumerate(problem):
                if problem_i == 20:
                # if problem_i != 0:
                    continue
                # print()
                print(problem_el.objective_function.name)

                opts = []
                for n_DoE in range(DoE_no):
                    exp_name = 'opt_data/' + exp_dict[model_type_el] + '/' + opt_problem_name + '/' \
                              + problem_el.objective_function.name + '/' + model_type_el + '/' + str(n_DoE)
                    hist_csv = exp_name + '.csv'
                    df_hist = pd.read_csv(hist_csv, index_col=0)
                    # print(df_hist['y - y_min'])
                    rec_csv = exp_name + '_rec.csv'
                    df_rec = pd.read_csv(rec_csv, index_col=0)
                    opt = df_rec['y - y_min'].values
                    print(opt)
                    opts.append(opt)
                med = np.median(opts) + 1e-3
                print()
                print(med)
                meds.append(med)
            meds_model.append(np.array(meds))
            # plt.hist(np.log(meds), ec='k', bins=30, label=model_type_el)
            # plt.legend()
        meds_diff = meds_model[1] / meds_model[0]
        print()
        print(meds_diff)
        plt.hist(np.log10(meds_diff), ec='k', bins=30)

stop = time.time()
# print('Total run time', stop - start)

plt.show()