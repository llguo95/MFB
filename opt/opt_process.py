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

# dim = 2
# noise_type = 'b'
# cost_ratio = 25
#
# # print()
# # print('dim = ', dim)
# # print('noise_type = ', noise_type)
# # print('cost_ratio = ', cost_ratio)
#
# problem = [
#     MFProblem(
#         objective_function=AugmentedTestFunction(
#             botorch_TestFunction(
#                 f(d=dim),
#             ), noise_type=noise_type, negate=False, # Minimization
#         ).to(**tkwargs),
#         cost_ratio=cost_ratio
#     )
#     for f in fs
# ]
#
# model_type = ['sogpr', 'stmf']
# lf = [.9]
# n_reg = [5 ** dim]
# n_reg_lf = [cost_ratio * 5 ** (dim - 1)]
# scramble = 1
# noise_fix = 0
# budget = 5 * 3 ** dim
# post_processing = 0
# acq_type = 'EI'
# iter_thresh = 25 * 3 ** (dim - 1)
# dev = 0
# # opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type
# DoE_no = 10

start = time.time()

# exp_dict = {
#     'sogpr': 'exp4_sogpr',
#     'stmf': 'exp4',
# }

exp_name = 'exp6'

dict_file = open(exp_name + '.pkl', 'rb')
exp_dict = pickle.load(dict_file)

for cost_ratio in exp_dict['cost_ratios']:

    for noise_type in exp_dict['noise_types']:

        for acq_type in exp_dict['acq_types']:

            for problem_name in exp_dict['problem']:

                for model_type_el in exp_dict['model_type']:
                    if model_type_el == 'sogpr':
                        exp_name_el = exp_name + '_sogpr'
                    else:
                        exp_name_el = exp_name

                    for lf_el in exp_dict['lf']:

                        opt_problem_name = str(1) + '_' + noise_type + '_' + acq_type + '_cr' + str(
                            cost_ratio) + '_lf' + str(lf_el) + '_tol0.01'

                        for n_DoE in range(exp_dict['DoE_no']):
                            df_path = 'opt_data/' + exp_name_el + '/' + opt_problem_name + '/' \
                                      + problem_name + '/' + model_type_el + '/' + str(n_DoE) + '_rec.csv'
                            df = pd.read_csv(df_path, index_col=0)
                            print(problem_name, df)

# for noise_type in ['b']:
#
#     for acq_type in ['UCB']:
#
#         meds_model = []
#         for model_type_el in model_type:
#             # print(model_type_el)
#
#             if model_type_el == 'sogpr':
#                 opt_problem_name = str(dim) + '_b_' + acq_type
#             else:
#                 opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type
#                 print()
#                 print(opt_problem_name)
#
#             meds = []
#             for problem_i, problem_el in enumerate(problem):
#                 # if problem_i == 20:
#                 # # if problem_i != 16:
#                 #     continue
#                 # print()
#                 # print(problem_el.objective_function.name)
#
#                 opts = []
#                 for n_DoE in range(DoE_no):
#                     exp_name = 'opt_data/' + exp_dict[model_type_el] + '/' + opt_problem_name + '/' \
#                               + problem_el.objective_function.name + '/' + model_type_el + '/' + str(n_DoE)
#                     hist_csv = exp_name + '.csv'
#                     df_hist = pd.read_csv(hist_csv, index_col=0)
#                     # print(df_hist['y - y_min'])
#                     rec_csv = exp_name + '_rec.csv'
#                     df_rec = pd.read_csv(rec_csv, index_col=0)
#                     opt = df_rec['y - y_min'].values
#                     # if problem_i == 27:
#                     #     print(df_rec['x_0'].values, opt)
#                     opts.append(opt)
#
#                 med = np.median(opts)
#                 # print()
#                 # print(med)
#                 meds.append(med)
#
#             meds_model.append(np.array(meds))
#             # plt.hist(np.log(meds), ec='k', bins=30, label=model_type_el)
#             # plt.legend()
#
#         meds_diff = meds_model[0] - meds_model[1]
#         print('Single-fidelity better for', np.sum(meds_diff < 0), 'functions')
#         print('Multi-fidelity better for', np.sum(meds_diff > 0), 'functions')
#         meds_list = list(zip(meds_diff, [p.objective_function.name for i, p in enumerate(problem)]))
#         meds_list.sort(key=lambda x: x[0])
#         print(meds_list)
#         # print()
#         meds_diff.sort()
#         # print(meds_diff)
#         # print(np.median(meds_diff))
#         # plt.figure(num=noise_type + acq_type)
#         # plt.hist(meds_diff, ec='k', bins=30)
#
# stop = time.time()
# # print('Total run time', stop - start)

plt.show()