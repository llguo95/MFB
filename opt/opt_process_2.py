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
print()

dim = 2

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim),
            ), negate=False,
        ).to(**tkwargs)
    )
    for f in fs
]#[:15]

model_type = ['sogpr']
lf = [.9]
noise_types = ['b']
acq_types = ['UCB']
cost_ratios = [10]
n_reg = [6 * 5 ** (dim - 1)]
scramble = 1
noise_fix = 0
budget = 50 # 5 * 3 ** (dim - 1)
tol = 0.005

print()
print('dim = ', dim)

dev = 1
DoE_no = 10
exp_name = 'exp_challenge'
vis_opt = 0

start = time.time()

# exp_name = 'exp6'

# dict_file = open(exp_name + '.pkl', 'rb')
exp_dict = dict(
    dim=dim,
    problem=[p.objective_function.name for p in problem],
    model_type=model_type,
    lf=lf,
    noise_types=noise_types,
    acq_types=acq_types,
    cost_ratios=cost_ratios,
    n_reg=n_reg,
    scramble=scramble,
    noise_fix=noise_fix,
    budget=budget,
    dev=dev,
    DoE_no=DoE_no,
    exp_name=exp_name,
    vis_opt=vis_opt,
    tol=tol,
)

for cost_ratio in exp_dict['cost_ratios']:

    for noise_type in exp_dict['noise_types']:

        for acq_type in exp_dict['acq_types']:

            pemeds = dict(
                sogpr=[], stmf=[],
            )

            ocmeds = dict(
                sogpr=[], stmf=[],
            )

            for model_type_el in exp_dict['model_type']:

                for problem_name in exp_dict['problem']:
                    print()
                    print(problem_name, model_type_el)
                    if model_type_el == 'sogpr':
                        exp_name_el = exp_name + '_sogpr'
                    else:
                        exp_name_el = exp_name

                    for lf_el in exp_dict['lf']:

                        opt_problem_name = str(exp_dict['dim']) + '_' + noise_type + '_' + acq_type + '_cr' + str(
                            cost_ratio) + '_lf' + str(lf_el)

                        pevals = []
                        ocvals = []
                        for n_DoE in range(exp_dict['DoE_no']):
                            df_path = 'opt_data/' + exp_name_el + '/' + opt_problem_name + '/' \
                                      + problem_name + '/' + model_type_el + '/' + str(n_DoE) + '_rec.csv'
                            df = pd.read_csv(df_path, index_col=0)
                            print(df['% error'].values, df['opt cost'].values)
                            pevals.append(df['% error'].values)
                            ocvals.append(df['opt cost'].values)

                        pemed = np.median(pevals)
                        ocmed = np.median(ocvals)
                        # print(pemed, ocmed)
                        pemeds[model_type_el].append(pemed)
                        ocmeds[model_type_el].append(ocmed)

                print()

                print(model_type_el)#, np.dot(pemeds[model_type_el], ocmeds[model_type_el]))

                plt.figure('pemed')
                plt.hist(np.log10([p + 1e-6 for p in pemeds[model_type_el]]), ec='k', bins=30, label=model_type_el, alpha=.5)
                print(pemeds[model_type_el])
                print(np.median(pemeds[model_type_el]))
                plt.legend()
                plt.figure('ocmed')
                plt.hist(ocmeds[model_type_el], ec='k', bins=30, label=model_type_el, alpha=.5)
                print(ocmeds[model_type_el])
                print(np.median(ocmeds[model_type_el]))
                plt.legend()

# print()
# print(pemeds)
# print(ocmeds)

plt.show()