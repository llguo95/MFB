import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybenchfunction

from matplotlib import colors

folder_path = 'data/'

# file_names_OLD = [
#     # dim 1
#     # '20220218110629', # mf, 5LF b
#     # '20220218172924', # mf, 5LF bn
#     # '20220222193253', # mf, 15LF b
#     # '20220225131321', # mf, 15LF b, RBF test (mtask)
#     # '20220222174825', # mf, 15LF bn
#     # '20220211165001', # mf, 30LF b
#     # '20220218144547', # mf, 30LF b, matern test (cokg)
#     # '20220218180630', # mf, 30LF bn
#     # '20220222132448', # mf, 30LF bn, matern test (cokg)
#     # '20220322145732_1d_b_nf_0_m', # mf, b, noise UNFIXED
#     # '20220322144906', # mf, b, noise FIXED
#
#     # dim 2
#     # '20220321173235', # mf, 25LF b
#     # '20220222143107', # mf, 75LF bn
#     # '20220222185639', # mf, 75LF b
#     # '20220323024607_2d_b_nf_0_m', # mf, b, noise UNFIXED
#
#     # dim 1
#     # '20220211170131', # sogpr, 6HF
#     # '20220322112515',  # sogpr, 5HF
#     # '20220322174532_1d_b_nf_0_s', # sogpr, 6HF noise UNFIXED
#
#     # '20220321140252', # sogpr, 6HF noise FIXED
#     # '20220221174116', # sogpr, 7HF
#     # '20220221174326', # sogpr, 8HF
#
#     # dim 2
#     # '20220321141651', # sogpr 25HF
#     # '20220222144503', # sogpr 30HF noise UNFIXED
#
#     # '20220322180144_2d_b_nf_0_s' # sogpr 30HF
# ]

# file_names = [
#     ### MF DATA ###
#     # dim 1
#     '20220322145732_1d_b_nf_0_m', # mf, b, noise UNFIXED
#     # '20220324171126_1d_bn_nf_0_m', # mf, bn, noise UNFIXED
#
#     # dim 2
#     # '20220323024607_2d_b_nf_0_m', # mf, b, noise UNFIXED
#     # '20220324183325_2d_bn_nf_0_m', # mf, bn, noise UNFIXED
#
#     # dim 3
#     # '20220326111937_3d_bn_nf_0_m', # mf, bn, noise UNFIXED
#     # '20220327202931_3d_b_nf_0_m', # mf, b, noise UNFIXED
#
#     ### SF DATA ###
#     # dim 1
#     '20220322174532_1d_b_nf_0_s', # sogpr, 6HF, noise UNFIXED
#
#     # dim 2
#     # '20220322180144_2d_b_nf_0_s', # sogpr 30HF, noise UNFIXED
#
#     # dim 3
#     # '20220323102210_3d_b_nf_0_s', # sogpr 150HF, noise UNFIXED
# ]

files_dict = {
    1: {
        'b': '20220322145732_1d_b_nf_0_m',
        'n': '20220331164922_1d_n_nf_0_m',
        # 'bn': '20220324171126_1d_bn_nf_0_m',
        'bn': '20220328214739_1d_bn_nf_0_m',
        'sogpr': '20220322174532_1d_b_nf_0_s',
    },
    2: {
        'b': '20220323024607_2d_b_nf_0_m',
        # 'bn': '20220324183325_2d_bn_nf_0_m',
        'bn': '20220328225544_2d_bn_nf_0_m',
        'sogpr': '20220322180144_2d_b_nf_0_s',
        'bn_exp': '20220404161017_2d_bn_nf_0_m',
    },
    3: {
        # 'b': '20220327202931_3d_b_nf_0_m',
        'b': '20220401022805_3d_b_nf_0_m',
        # 'bn': '20220326111937_3d_bn_nf_0_m',
        'bn': '20220331103524_3d_bn_nf_0_m',
        'sogpr': '20220323102210_3d_b_nf_0_s',
    }
}

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

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]
# print([f.name for f in fs])

# model_types = ['cokg', 'cokg_dms', 'mtask']
# lfs = [0.1, 0.5, 0.9]

# mrmv = []
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
for dim_i, dim in enumerate([1, 2, 3]):
    # if dim != 2: continue
    files_dict_dim = files_dict[dim]

    # print(dim)

    noise_type_data = {}
    RAAE_medians_dict_noisetype = {}
    for noise_type_i, noise_type in enumerate(['b', 'n', 'bn', 'sogpr', 'bn_exp']):
        if noise_type == 'n' and dim != 1: continue
        if noise_type == 'bn_exp' and dim != 2: continue
        files_dict_noise_type = files_dict_dim[noise_type]

        open_file = open(folder_path + files_dict_noise_type + '.pkl', 'rb')
        data = pickle.load(open_file)
        open_file.close()

        open_file = open(folder_path + files_dict_noise_type + '_metadata.pkl', 'rb')
        metadata = pickle.load(open_file)
        processed_metadata[(dim, noise_type)] = metadata

        open_file.close()

        # print(noise_type)

        model_type_data = {}
        RAAE_medians_dict_modeltype = {}
        for model_no, model_type in enumerate(metadata['model_type']):
            model_type_slice = data[model_type]

            # print(model_type)

            lf_data = {}
            RAAE_medians_dict_lf = {}
            for lf in metadata['lf']:

                # print(lf)

                problem_data = {}
                for problem_no, problem in enumerate(metadata['problem']):
                    # if problem == 'AugmentedRidge' and metadata['dim'] == 1: continue

                    problem_slice = model_type_slice[problem]
                    lf_slice = problem_slice[lf]

                    n_reg_data = []
                    for n_reg, n_reg_lf in zip(metadata['n_reg'], metadata['n_reg_lf']):
                        n_reg_slice = lf_slice[(n_reg, n_reg_lf)]

                        n_reg_data.append(n_reg_slice['RAAE_stats']['median'])

                    problem_data[problem] = n_reg_data

                RAAE_medians = np.array(list(problem_data.values()))

                RAAE_medians_dict_lf[lf] = RAAE_medians

                # lf_data[lf] = RAAE_medians
                lf_data[lf] = problem_data

            RAAE_medians_dict_modeltype[model_type] = RAAE_medians_dict_lf
            model_type_data[model_type] = lf_data
            # print(RAAE_medians_dict)

        RAAE_medians_dict_noisetype[noise_type] = RAAE_medians_dict_modeltype
        noise_type_data[noise_type] = model_type_data

    processed_data_mf[dim] = noise_type_data

    for noise_type_i, noise_type in enumerate(['b', 'n', 'bn', 'bn_exp']):
        if noise_type == 'n' and dim != 1: continue
        if noise_type == 'bn_exp' and dim != 2: continue
        metadata = processed_metadata[(dim, noise_type)]
        # print(noise_type)
        for model_no, model_type in enumerate(metadata['model_type']):
            # print(model_type)
            for lf_i, lf in enumerate(metadata['lf']):
                # print(lf)

                if noise_type == 'b' and lf == 0.5 and model_type == 'cokg':
                    RAAEs_sogpr_dim.append(RAAE_medians_dict_noisetype['sogpr']['sogpr'][0.5].flatten())

                if show_sogpr_dim and model_type == 'cokg_dms' and noise_type == 'b' and dim == 1:
                    plt.figure(num=str(dim) + '_' + noise_type + '_' + model_type + '_' + str(lf) + '_' + 'individual')
                    plt.boxplot(RAAE_medians_dict_noisetype[noise_type][model_type][lf])
                    plt.xticks(range(1, 6), [(2 * k + 1) * 5 ** dim for k in range(5)])
                    plt.grid()

                ratio_noisetype_modeltype = RAAE_medians_dict_noisetype['sogpr']['sogpr'][0.5] / RAAE_medians_dict_noisetype[noise_type][model_type][lf]
                log_ratio_noisetype_modeltype = np.log10(ratio_noisetype_modeltype)
                # print(np.median(log_ratio_noisetype_modeltype, axis=0))

                log_ratio_grids[(noise_type, model_type, lf, dim)] = np.median(log_ratio_noisetype_modeltype, axis=0)

                # if show_histograms and dim == 1 and noise_type == 'b' and lf == 0.5 and model_type == 'cokg_dms':
                if show_histograms and dim == 2 and noise_type != 'bn_exp':
                    prop, nonprop = [], []
                    k = 2
                    for stat_i, stat in enumerate(log_ratio_noisetype_modeltype[:, k]):
                        (nonprop, prop)[fs[stat_i].multimodal is False].append(stat)
                    # print(dim, model_type, lf, noise_type, np.median(log_ratio_noisetype_modeltype[:, 3]))
                    # print('prop', prop, 'nonprop', nonprop)

                    if lf_i == 0 and model_no == 0:
                        fig, axs = plt.subplots(nrows=3, ncols=3, num=str(dim) + '_' + noise_type + '_' + str((2 * k + 1) * 5 ** dim), figsize=(8, 8), sharex='all', sharey='all',)
                        plt.suptitle(str(dim) + 'D, ' + noise_type_dict[noise_type] + ', ' + str((2 * k + 1) * 5 ** dim) + ' low fid. pts.')

                    ax = axs[model_no, lf_i]
                    if model_no == 0:
                        ax.set_title('LF = ' + str(lf))
                    # plt.hist(log_ratio_noisetype_modeltype[:, 2], bins=20, ec='k')
                    ax.hist((prop, nonprop), bins=10, ec='k', color=['g', 'r'], stacked=True)
                    # print('prop', np.median(prop), 'nonprop', np.median(nonprop))
                    ax.axvline(x=0, color='k', linewidth=2, label='improvement threshold')
                    ax.axvline(x=np.median(prop), linestyle='--', color='darkgreen', linewidth=3, label='convex med. (' + str(np.round(np.median(prop), 2)) + ')')
                    ax.axvline(x=np.median(nonprop), linestyle='--', color='darkred', linewidth=3, label='nonconvex med. (' + str(np.round(np.median(nonprop), 2)) + ')')
                    if model_no == 2:
                        ax.set_xlabel('Log relative improvement')
                    if lf_i == 0:
                        ax.set_ylabel('Frequency')
                    if lf_i == 2:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(model_names_dict[model_type])
                    # plt.grid()
                    ax.legend(prop={'size': 9})
                    plt.tight_layout()

                if show_boxplots and noise_type != 'bn_exp':
                    if lf_i == 0 and model_no == 0:
                        fig, axs = plt.subplots(nrows=3, ncols=3, num=str(dim) + '_' + noise_type, figsize=(8, 8), sharey='all')
                        plt.suptitle(str(dim) + 'D, ' + noise_type_dict[noise_type])
                    ax = axs[model_no, lf_i]
                    ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=.5)
                    ax.boxplot(log_ratio_noisetype_modeltype)
                    if model_no == 0:
                        ax.set_title('LF = ' + str(lf))
                    if model_no == 2:
                        ax.set_xlabel('No. of LF data points')
                    if lf_i == 0:
                        ax.set_ylabel('Log relative improvement')
                    if lf_i == 2:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(model_names_dict[model_type])
                    ax.set_xticks(range(1, 6), [(2 * k + 1) * 5 ** dim for k in range(5)])
                    ax.grid()
                    plt.tight_layout()

if show_sogpr_dim:
    plt.figure(num='sogpr')
    plt.boxplot(np.array(RAAEs_sogpr_dim).T, showfliers=False)

# print(np.amin(list(log_ratio_grids.values())))
# print(np.quantile(list(log_ratio_grids.values()), q=.1))
# print(np.quantile(list(log_ratio_grids.values()), q=.9))
# print(np.amax(list(log_ratio_grids.values())))

if show_contours:

    min_lr, max_lr = np.amin(list(log_ratio_grids.values())), np.amax(list(log_ratio_grids.values()))
    min_cm, max_cm = min_lr, 1

    divnorm2 = colors.TwoSlopeNorm(vmin=min_cm, vcenter=0., vmax=max_cm)
    cmap = colors.ListedColormap(['b', 'r', 'r'])
    divnorm = colors.BoundaryNorm([min_lr, min_cm, max_cm, max_lr], cmap.N)

    for noise_type_i, noise_type in enumerate(['b', 'bn']):
        fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
        # print(noise_type)
        for model_no, model_type in enumerate(metadata['model_type']):
            # print(model_type)
            for lf_i, lf in enumerate(metadata['lf']):
                # print(lf)
                log_ratio_grid = np.array([log_ratio_grids[(noise_type, model_type, lf, k)] for k in range(1, 4)])
                # plt.figure(num=noise_type + '_' + model_type + '_' + str(lf) + '_' + 'contour')
                X, Y = np.meshgrid([5, 15, 25, 35, 45], [1, 2, 3])
                ax = axs[lf_i, model_no]
                matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
                cf = ax.contourf(X, Y, np.array(log_ratio_grid), levels=np.linspace(min_lr, max_lr, 50), cmap=cmap, norm=divnorm)
                cf2 = ax.contourf(X, Y, np.array(log_ratio_grid), levels=np.linspace(min_cm, max_cm, 50), cmap='bwr', norm=divnorm2)
                c = ax.contour(X, Y, np.array(log_ratio_grid), colors='k', alpha=.75)
                ax.clabel(c, inline=1, fontsize=6)
                if lf_i == 2:
                    ax.set_xlabel('cost ratio')
                if model_no == 0:
                    ax.set_ylabel('dim')
                ax.set_title(model_type + ', LF = ' + str(lf))

        plt.suptitle('noise type ' + noise_type)
        # cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        plt.tight_layout()
        # fig.colorbar(cf2, cax=cbar_ax)

# plt.tight_layout()
plt.show()