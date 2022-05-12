import pickle

import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood

import pybenchfunction

from matplotlib import colors

import os

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.multitask import MultiTaskGP

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

from MFproblem import MFProblem
from cokgj import CoKrigingGP
from objective_formatter import AugmentedTestFunction, botorch_TestFunction
from pipeline import trainer, posttrainer

reg_data_path = 'reg_data/'

model_names_dict = {
    'cokg': 'cokg-j',
    'cokg_dms': 'cokg-d',
    'mtask': 'mtask',
}

noise_type_dict = {
    'b': 'bias only',
    'n': 'noise only',
    'bn': 'bias and noise',
}

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]

# problem_list = [('Augmented' + f.name).replace(" ", "") for f in fs]
dim_list = [1, 2, 3]
noise_list = ['b', 'n', 'bn']
model_list = ['sogpr', 'cokg', 'cokg_dms', 'mtask']
lf_list = [0.1, 0.5, 0.9]
def n_reg_lf_list(dim):
    return [(3 * k + 1) * 5 ** dim for k in range(4)]

metadata_mf = None

processed_data_mf = {}
processed_metadata = {}

show_histograms = 1
show_boxplots = 0
show_contours = 0
show_sogpr_dim = 0

log_ratio_grids = {}
RAAEs_sogpr_dim = []
RAAEs_mogpr_dim = []

def process():
    for dim_i, dim in enumerate(dim_list):

        for noise_type_i, noise_type in enumerate(noise_list):

            problem_list = [
                MFProblem(
                    objective_function=AugmentedTestFunction(
                        botorch_TestFunction(
                            f(d=dim), negate=False,  # Minimization
                        ), noise_type=noise_type,
                    ).to(**tkwargs)
                )
                for f in fs
            ]

            for problem_no, problem in enumerate(problem_list):

                for lf_i, lf in enumerate(lf_list):

                    for n_reg, n_reg_lf in zip([5 ** dim] * len(n_reg_lf_list(dim)), n_reg_lf_list(dim)):

                        problem_folder_name = str(dim) + ',' \
                                              + noise_type + ',' \
                                              + '0,' \
                                              + problem.objective_function.name + ',' \
                                              + str(lf) + ',' \
                                              + str(n_reg) + ',' \
                                              + str(n_reg_lf)

                        problem_path = reg_data_path + problem_folder_name

                        if os.path.exists(problem_path):

                            for model_i, model_type in enumerate(model_list):

                                model_path = problem_path + '/' + model_type

                                if os.path.exists(model_path):

                                    for exp_i in range(10):

                                        exp_path = model_path + '/' + str(exp_i)
                                        print(exp_path)

                                        if os.path.exists(exp_path) and model_type != 'cokg_dms':

                                            state_dict = torch.load(exp_path)

                                            if os.path.exists(exp_path + '_train_x'):

                                                train_x = torch.load(exp_path + '_train_x')
                                                train_y = torch.load(exp_path + '_train_y')

                                                model = trainer(train_x, train_y, problem, model_type, dim,
                                                                0, 1e-4, optimize=False)

                                                # print('PRE LIKELIHOOD', model.state_dict())
                                                # if noise_fix:
                                                #     model.likelihood = GaussianLikelihood(
                                                #         noise_constraint=gpytorch.constraints.Interval(1e-4, 2e-4))
                                                #     # cons = gpytorch.constraints.constraints.Interval(1e-4, 2e-4)
                                                #     # model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                                                # else:
                                                model.likelihood = GaussianLikelihood(
                                                    noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
                                                # print('POST LIKELIHOOD', model.state_dict())

                                                # (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
                                                #  test_y_var_list_low,) = posttrainer(
                                                #     model, model_type, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low
                                                # )

                                                if exp_i == 0: return

process()

#                     problem_data[problem] = n_reg_data
#
#                 RAAE_medians = np.array(list(problem_data.values()))
#
#                 RAAE_medians_dict_lf[lf] = RAAE_medians
#
#                 # lf_data[lf] = RAAE_medians
#                 lf_data[lf] = problem_data
#
#             RAAE_medians_dict_modeltype[model_type] = RAAE_medians_dict_lf
#             model_type_data[model_type] = lf_data
#             # print(RAAE_medians_dict)
#
#         RAAE_medians_dict_noisetype[noise_type] = RAAE_medians_dict_modeltype
#         noise_type_data[noise_type] = model_type_data
#
#     processed_data_mf[dim] = noise_type_data
#
#     for noise_type_i, noise_type in enumerate(['b', 'n', 'bn', 'bn_exp']):
#         if noise_type == 'n' and dim != 1: continue
#         if noise_type == 'bn_exp' and dim != 2: continue
#         metadata = processed_metadata[(dim, noise_type)]
#         # print(noise_type)
#         for model_no, model_type in enumerate(metadata['model_type']):
#             # print(model_type)
#             for lf_i, lf in enumerate(metadata['lf']):
#                 # print(lf)
#
#                 if noise_type == 'b' and lf == 0.5 and model_type == 'cokg':
#                     RAAEs_sogpr_dim.append(RAAE_medians_dict_noisetype['sogpr']['sogpr'][0.5].flatten())
#
#                 if show_sogpr_dim and model_type == 'cokg_dms' and noise_type == 'b' and dim == 1:
#                     plt.figure(num=str(dim) + '_' + noise_type + '_' + model_type + '_' + str(lf) + '_' + 'individual')
#                     plt.boxplot(RAAE_medians_dict_noisetype[noise_type][model_type][lf])
#                     plt.xticks(range(1, 6), [(2 * k + 1) * 5 ** dim for k in range(5)])
#                     plt.grid()
#
#                 ratio_noisetype_modeltype = RAAE_medians_dict_noisetype['sogpr']['sogpr'][0.5] / RAAE_medians_dict_noisetype[noise_type][model_type][lf]
#                 log_ratio_noisetype_modeltype = np.log10(ratio_noisetype_modeltype)
#                 # print(np.median(log_ratio_noisetype_modeltype, axis=0))
#
#                 log_ratio_grids[(noise_type, model_type, lf, dim)] = np.median(log_ratio_noisetype_modeltype, axis=0)
#
#                 # if show_histograms and dim == 1 and noise_type == 'b' and lf == 0.5 and model_type == 'cokg_dms':
#                 if show_histograms and dim == 2 and noise_type != 'bn_exp':
#                     prop, nonprop = [], []
#                     k = 2
#                     for stat_i, stat in enumerate(log_ratio_noisetype_modeltype[:, k]):
#                         (nonprop, prop)[fs[stat_i].multimodal is False].append(stat)
#                     # print(dim, model_type, lf, noise_type, np.median(log_ratio_noisetype_modeltype[:, 3]))
#                     # print('prop', prop, 'nonprop', nonprop)
#
#                     if lf_i == 0 and model_no == 0:
#                         fig, axs = plt.subplots(nrows=3, ncols=3, num=str(dim) + '_' + noise_type + '_' + str((2 * k + 1) * 5 ** dim), figsize=(8, 8), sharex='all', sharey='all',)
#                         plt.suptitle(str(dim) + 'D, ' + noise_type_dict[noise_type] + ', ' + str((2 * k + 1) * 5 ** dim) + ' low fid. pts.')
#
#                     ax = axs[model_no, lf_i]
#                     if model_no == 0:
#                         ax.set_title('LF = ' + str(lf))
#                     # plt.hist(log_ratio_noisetype_modeltype[:, 2], bins=20, ec='k')
#                     ax.hist((prop, nonprop), bins=10, ec='k', color=['g', 'r'], stacked=True)
#                     # print('prop', np.median(prop), 'nonprop', np.median(nonprop))
#                     ax.axvline(x=0, color='k', linewidth=2, label='improvement threshold')
#                     ax.axvline(x=np.median(prop), linestyle='--', color='darkgreen', linewidth=3, label='convex med. (' + str(np.round(np.median(prop), 2)) + ')')
#                     ax.axvline(x=np.median(nonprop), linestyle='--', color='darkred', linewidth=3, label='nonconvex med. (' + str(np.round(np.median(nonprop), 2)) + ')')
#                     if model_no == 2:
#                         ax.set_xlabel('Log relative improvement')
#                     if lf_i == 0:
#                         ax.set_ylabel('Frequency')
#                     if lf_i == 2:
#                         ax.yaxis.set_label_position("right")
#                         ax.set_ylabel(model_names_dict[model_type])
#                     # plt.grid()
#                     ax.legend(prop={'size': 9})
#                     plt.tight_layout()
#
#                 if show_boxplots and noise_type != 'bn_exp':
#                     if lf_i == 0 and model_no == 0:
#                         fig, axs = plt.subplots(nrows=3, ncols=3, num=str(dim) + '_' + noise_type, figsize=(8, 8), sharey='all')
#                         plt.suptitle(str(dim) + 'D, ' + noise_type_dict[noise_type])
#                     ax = axs[model_no, lf_i]
#                     ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=.5)
#                     ax.boxplot(log_ratio_noisetype_modeltype)
#                     if model_no == 0:
#                         ax.set_title('LF = ' + str(lf))
#                     if model_no == 2:
#                         ax.set_xlabel('No. of LF data points')
#                     if lf_i == 0:
#                         ax.set_ylabel('Log relative improvement')
#                     if lf_i == 2:
#                         ax.yaxis.set_label_position("right")
#                         ax.set_ylabel(model_names_dict[model_type])
#                     ax.set_xticks(range(1, 6), [(2 * k + 1) * 5 ** dim for k in range(5)])
#                     ax.grid()
#                     plt.tight_layout()
#
# if show_sogpr_dim:
#     plt.figure(num='sogpr')
#     plt.boxplot(np.array(RAAEs_sogpr_dim).T, showfliers=False)
#
# # print(np.amin(list(log_ratio_grids.values())))
# # print(np.quantile(list(log_ratio_grids.values()), q=.1))
# # print(np.quantile(list(log_ratio_grids.values()), q=.9))
# # print(np.amax(list(log_ratio_grids.values())))
#
# if show_contours:
#
#     min_lr, max_lr = np.amin(list(log_ratio_grids.values())), np.amax(list(log_ratio_grids.values()))
#     min_cm, max_cm = min_lr, 1
#
#     divnorm2 = colors.TwoSlopeNorm(vmin=min_cm, vcenter=0., vmax=max_cm)
#     cmap = colors.ListedColormap(['b', 'r', 'r'])
#     divnorm = colors.BoundaryNorm([min_lr, min_cm, max_cm, max_lr], cmap.N)
#
#     for noise_type_i, noise_type in enumerate(['b', 'bn']):
#         fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
#         # print(noise_type)
#         for model_no, model_type in enumerate(metadata['model_type']):
#             # print(model_type)
#             for lf_i, lf in enumerate(metadata['lf']):
#                 # print(lf)
#                 log_ratio_grid = np.array([log_ratio_grids[(noise_type, model_type, lf, k)] for k in range(1, 4)])
#                 # plt.figure(num=noise_type + '_' + model_type + '_' + str(lf) + '_' + 'contour')
#                 X, Y = np.meshgrid([5, 15, 25, 35, 45], [1, 2, 3])
#                 ax = axs[lf_i, model_no]
#                 matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#                 cf = ax.contourf(X, Y, np.array(log_ratio_grid), levels=np.linspace(min_lr, max_lr, 50), cmap=cmap, norm=divnorm)
#                 cf2 = ax.contourf(X, Y, np.array(log_ratio_grid), levels=np.linspace(min_cm, max_cm, 50), cmap='bwr', norm=divnorm2)
#                 c = ax.contour(X, Y, np.array(log_ratio_grid), colors='k', alpha=.75)
#                 ax.clabel(c, inline=1, fontsize=6)
#                 if lf_i == 2:
#                     ax.set_xlabel('cost ratio')
#                 if model_no == 0:
#                     ax.set_ylabel('dim')
#                 ax.set_title(model_type + ', LF = ' + str(lf))
#
#         plt.suptitle('noise type ' + noise_type)
#         # cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
#         plt.tight_layout()
#         # fig.colorbar(cf2, cax=cbar_ax)
#
# # plt.tight_layout()
# plt.show()