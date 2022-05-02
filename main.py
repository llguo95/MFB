import pickle

import GPy
import numpy as np
import torch
from botorch.models import SingleTaskGP
from matplotlib import pyplot as plt
from scipy.stats import norm

from gpytorch.constraints.constraints import Interval

import os

from botorch import fit_gpytorch_model
from pybenchfunction import function

from scipy.stats.mstats import gmean

from sklearn.metrics import r2_score, mean_tweedie_deviance

from objective_formatter import botorch_TestFunction, AugmentedTestFunction

from MFproblem import MFProblem
from pipeline import pretrainer, trainer, posttrainer, reg_main_visualizer

import time

np.random.seed(None)

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}


def ei(mean_x, var_x, f_inc):
    mean_x = -mean_x  # minimization
    Delta = mean_x - f_inc
    std = np.sqrt(np.abs(var_x))
    res = np.maximum(Delta, np.zeros(Delta.shape)) \
          + std * norm.pdf(Delta / std) \
          - np.abs(Delta) * norm.cdf(Delta / std)
    return res


def ucb(mean_x, var_x, kappa):
    mean_x = -mean_x  # minimization
    res = mean_x - kappa * np.sqrt(np.abs(var_x))
    return res


def uniform_grid(bl, tr, n, mesh=False):
    coord_axes = [np.linspace(bl_c, tr_c, n_c) for (bl_c, tr_c, n_c) in zip(bl, tr, n)]
    coord_mesh = np.array(np.meshgrid(*coord_axes))
    s = coord_mesh.shape
    coord_list = coord_mesh.reshape((s[0], np.prod(s[1:]))).T
    if mesh:
        res = coord_mesh, coord_list
    else:
        res = coord_list
    return res


vis = 0


# def reg_main_old(
#         problem=None, model_type=None, lf=None, n_reg=None, n_reg_lf=None, scramble=False, random=False,
#         n_inf=500, noise_fix=True, lf_jitter=1e-4, noise_type='b',
# ):
#     if problem is None:
#         problem = [
#             MFProblem(
#                 objective_function=AugmentedTestFunction(
#                     botorch_TestFunction(
#                         function.Ackley(d=1), negate=False
#                     )
#                 ).to(**tkwargs)
#             ),
#         ]
#
#     if lf is None:
#         lf = [0.5]
#
#     if model_type is None:
#         model_type = ['cokg']
#
#     if n_reg is None:
#         n_reg = [5]
#
#     if n_reg_lf is None:
#         n_reg_lf = [10]
#
#     if scramble:
#         n_DoE = 10
#     else:
#         n_DoE = 1
#
#     metadata_dict = {
#         'problem': [p.objective_function.name for p in problem],
#         'model_type': model_type,
#         'lf': lf,
#         'n_reg': n_reg,
#         'n_reg_lf': n_reg_lf,
#         'scramble': scramble,
#         'noise_fix': noise_fix,
#         'noise_type': noise_type,
#     }
#
#     model_type_data = {}
#     for model_type_el in model_type:
#         print()
#         print(model_type_el)
#
#         problem_data = {}
#         for problem_el in problem:
#             print()
#             print(problem_el.objective_function.name)
#
#             bds = problem_el.bounds
#             dim = problem_el.objective_function.dim
#
#             lf_data = {}
#             for lf_el in lf:
#                 print()
#                 print('lf =', lf_el)
#
#                 n_reg_data = {}
#                 RAAE_stats_dict = {}
#                 RMSTD_stats_dict = {}
#                 # ei_stats_dict = {}
#                 # ucb_stats_dict = {}
#                 for n_reg_el, n_reg_lf_el in zip(n_reg, n_reg_lf):
#                     print('n_reg =', n_reg_el)
#                     print('n_reg_lf_el =', n_reg_lf_el)
#
#                     n_DoE_RAAE_data = []
#                     n_DoE_RMSTD_data = []
#                     # n_DoE_x_rec_data = {
#                     #     'ei': [],
#                     #     'ucb': [],
#                     # }
#                     # n_DoE_y_rec_data = {
#                     #     'ei': [],
#                     #     'ucb': [],
#                     # }
#
#                     for _ in range(n_DoE):
#                         ####################
#                         ### Pre-training ###
#                         ####################
#
#                         (train_x, train_y_high, train_obj, test_x_list, test_x_list_scaled, test_x_list_high,
#                          scaler_y_high, exact_y, exact_y_low, train_y_low, scaler_y_low, ) = pretrainer(
#                             problem_el, model_type_el, n_reg_el, n_reg_lf_el, lf_el, random, scramble, n_inf, bds, dim,
#                         )
#
#                         ################
#                         ### Training ###
#                         ################
#
#                         model = trainer(train_x, train_obj, problem_el, model_type_el, dim, noise_fix, lf_jitter, )
#
#                         #################################
#                         ### Post-training; prediction ###
#                         #################################
#
#                         (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
#                          test_y_var_list_low,) = posttrainer(
#                             model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low
#                         )
#
#                         ########################
#                         ### Post-processing ####
#                         ########################
#
#                         if model_type_el != 'cokg_dms':
#                             exact_y_scaled = model.outcome_transform(torch.tensor(exact_y))[0].detach().numpy()[0]
#                         else:
#                             exact_y_scaled = scaler_y_high.transform(exact_y[:, None])
#
#                         pred_diff = exact_y - test_y_list_high.flatten()
#                         pred_diff_scaled = exact_y_scaled - test_y_list_high_scaled.flatten()
#
#                         L1_PE = np.abs(pred_diff).sum()  # L1 prediction error
#                         L2_PE = np.sqrt((pred_diff ** 2).sum())  # L2 prediction error
#
#                         L1_SD = np.abs(exact_y - np.mean(exact_y)).sum()  # L1 sample deviation
#                         L2_SD = np.sqrt(((exact_y - np.mean(exact_y)) ** 2).sum())  # L2 sample deviation
#
#                         # print(L1_PE, L2_PE)
#
#                         RAAE = 1 / np.sqrt(len(pred_diff)) * L1_PE / L2_SD
#
#                         SL1 = np.mean(np.abs(pred_diff)) / np.std(exact_y)  # relative average absolute error
#                         print(RAAE, SL1)
#
#                         # SL1_scaled = np.mean(np.abs(pred_diff_scaled)) / np.std(exact_y_scaled)
#                         #
#                         # RMAE = np.median(np.abs(pred_diff)) / np.std(exact_y) # relative median absolute error
#                         # RMAE_scaled = np.median(np.abs(pred_diff_scaled)) / np.std(exact_y_scaled)
#                         #
#                         # SL2 = np.sqrt(np.mean(pred_diff ** 2)) / np.std(exact_y) # square root relative average variance
#                         # SL2_scaled = np.sqrt(np.mean(pred_diff_scaled ** 2)) / np.std(exact_y_scaled)
#                         #
#                         # SRRMV = np.sqrt(np.median(pred_diff ** 2)) / np.std(exact_y) # square root relative median variance
#                         # SRRMV_scaled = np.sqrt(np.median(pred_diff_scaled ** 2)) / np.std(exact_y_scaled)
#                         #
#                         # L1 = np.median(np.abs(pred_diff)) # mean absolute error (L1 error)
#                         # L1_scaled = np.median(np.abs(pred_diff_scaled))
#                         #
#                         # L2 = np.sqrt(np.mean(pred_diff ** 2)) # square root-mean-square error (L2 error)
#                         # L2_scaled = np.sqrt(np.mean(pred_diff_scaled ** 2))
#                         #
#                         # R2 = r2_score(exact_y, test_y_list_high.flatten())
#                         # R2_scaled = r2_score(exact_y_scaled, test_y_list_high_scaled.flatten())
#                         #
#                         # print('SL1 ', SL1, SL1_scaled)
#                         # print('SL2', SL2, SL2_scaled)
#                         # print('RMAE ', RMAE, RMAE_scaled)
#                         # print('R2   ', R2, R2_scaled)
#                         # print('L1  ', L1, L1_scaled)
#                         # print('L2', L2, L2_scaled)
#
#                         RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
#
#                         # acq_ei = ei(
#                         #     mean_x=test_y_list_high,
#                         #     var_x=test_y_var_list_high,
#                         #     f_inc=np.amax(train_y_high)
#                         # )
#                         #
#                         # acq_ucb = ucb(
#                         #     mean_x=test_y_list_high,
#                         #     var_x=test_y_var_list_high,
#                         #     kappa=2
#                         # )
#
#                         # if max(acq_ei) != min(acq_ei):
#                         #     acq_ei_norm = (acq_ei - min(acq_ei)) / (max(acq_ei) - min(acq_ei))
#                         # else:
#                         #     acq_ei_norm = acq_ei
#                         #
#                         # if max(acq_ucb) != min(acq_ucb):
#                         #     acq_ucb_norm = (acq_ucb - min(acq_ucb)) / (max(acq_ucb) - min(acq_ucb))
#                         # else:
#                         #     acq_ucb_norm = acq_ucb
#
#                         # if vis:
#                         #     reg_main_visualizer(_, test_x_list, test_y_list_high, test_y_var_list_high, exact_y, acq_ei_norm, acq_ucb_norm)
#
#                         # x_next_ei = test_x_list[np.argmax(acq_ei_norm)]
#                         # y_next_ei = problem_el.objective_function(torch.tensor([x_next_ei[0], 1])).cpu().detach().numpy()
#
#                         # x_next_ucb = test_x_list[np.argmax(acq_ucb_norm)]
#                         # y_next_ucb = problem_el.objective_function(torch.tensor([x_next_ucb[0], 1])).cpu().detach().numpy()
#
#                         n_DoE_RAAE_data.append(L1_PE)
#                         n_DoE_RMSTD_data.append(RMSTD)
#
#                         x_opt, y_opt = problem_el.objective_function.opt(d=dim - 1)
#
#                         # n_DoE_x_rec_data['ei'].append(x_next_ei)
#                         # n_DoE_x_rec_data['ucb'].append(x_next_ucb)
#
#                         # n_DoE_y_rec_data['ei'].append(np.abs(y_next_ei - y_opt) / (np.amax(exact_y) - y_opt))
#                         # n_DoE_y_rec_data['ucb'].append(np.abs(y_next_ucb - y_opt) / (np.amax(exact_y) - y_opt))
#
#                         vis2d = True
#                         if vis2d:
#                             if dim - 1 == 2:
#                                 coord_mesh, _ = uniform_grid(bl=bds[0], tr=bds[1], n=[22, 22], mesh=True)
#                                 fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
#                                                        num=problem_el.objective_function.name)
#                                 ax.plot_surface(coord_mesh[0], coord_mesh[1],
#                                                 test_y_list_high.reshape(coord_mesh[0].shape), cmap='viridis',
#                                                 linewidth=0, alpha=.5)
#                                 ax.scatter(train_x[:n_reg_el][:, 0], train_x[:n_reg_el][:, 1], train_y_high, c='r',
#                                            s=50)
#                                 plt.tight_layout()
#
#                                 # print(model.covar_module.base_kernel)
#                             elif dim - 1 == 1:
#                                 plt.figure(num=problem_el.objective_function.name + '_' + model_type_el, figsize=(4, 4))
#                                 coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
#                                 plt.plot(coord_list, test_y_list_high, 'r--')
#                                 plt.fill_between(coord_list.flatten(),
#                                                  (test_y_list_high - 2 * np.sqrt(
#                                                      np.abs(test_y_var_list_high))).flatten(),
#                                                  (test_y_list_high + 2 * np.sqrt(
#                                                      np.abs(test_y_var_list_high))).flatten(),
#                                                  alpha=.25, color='r')
#                                 plt.plot(coord_list, exact_y, 'r', linewidth=.5)
#                                 if model_type_el != 'cokg_dms':
#                                     plt.scatter(train_x[:n_reg_el][:, 0], train_y_high, c='r', )
#
#                                 if model_type_el in ['cokg', 'cokg_dms', 'mtask']:
#                                     # plt.figure(num=problem_el.objective_function.name + '_lf', figsize=(4, 4))
#                                     plt.plot(coord_list, test_y_list_low, 'b--')
#                                     plt.fill_between(coord_list.flatten(),
#                                                      (test_y_list_low - 2 * np.sqrt(
#                                                          np.abs(test_y_var_list_low))).flatten(),
#                                                      (test_y_list_low + 2 * np.sqrt(
#                                                          np.abs(test_y_var_list_low))).flatten(),
#                                                      alpha=.25, color='b')
#                                     plt.plot(coord_list, exact_y_low, 'b', linewidth=.5)
#                                     if model_type_el != 'cokg_dms':
#                                         plt.scatter(train_x[n_reg_el:][:, 0], train_y_low, c='b', )
#
#                                 plt.grid()
#                                 plt.tight_layout()
#
#                                 # plt.figure(num=problem_el.objective_function.name + '_errdist')
#                                 # # plt.hist((exact_y - test_y_list_high).flatten(), ec='k', bins=20)
#                                 # plt.hist(np.abs(exact_y_scaled[0] - test_y_list_high_scaled.flatten()), ec='k', bins=20)
#                                 # plt.tight_layout()
#
#                     # RAAE_stats_dict['amean'] = np.mean(n_DoE_RAAE_data)
#                     # RAAE_stats_dict['gmean'] = gmean(n_DoE_RAAE_data)
#                     RAAE_stats_dict['median'] = np.median(n_DoE_RAAE_data)
#                     RAAE_stats_dict['quantiles'] = [np.quantile(n_DoE_RAAE_data, q=.25),
#                                                     np.quantile(n_DoE_RAAE_data, q=.75)]
#
#                     # RMSTD_stats_dict['amean'] = np.mean(n_DoE_RMSTD_data)
#                     # RMSTD_stats_dict['gmean'] = gmean(n_DoE_RMSTD_data)
#                     RMSTD_stats_dict['median'] = np.median(n_DoE_RMSTD_data)
#                     RMSTD_stats_dict['quantiles'] = [np.quantile(n_DoE_RMSTD_data, q=.25),
#                                                      np.quantile(n_DoE_RMSTD_data, q=.75)]
#
#                     # ei_stats_dict['median'] = np.median(n_DoE_y_rec_data['ei'])
#                     # ei_stats_dict['quantiles'] = [np.quantile(n_DoE_y_rec_data['ei'], q=.25), np.quantile(n_DoE_y_rec_data['ei'], q=.75)]
#                     #
#                     # ucb_stats_dict['median'] = np.median(n_DoE_y_rec_data['ucb'])
#                     # ucb_stats_dict['quantiles'] = [np.quantile(n_DoE_y_rec_data['ucb'], q=.25), np.quantile(n_DoE_y_rec_data['ucb'], q=.75)]
#
#                     n_reg_data[(n_reg_el, n_reg_lf_el)] = {
#                         'RAAE_stats': RAAE_stats_dict.copy(),
#                         'RMSTD_stats': RMSTD_stats_dict.copy(),
#                         # 'ei_stats': ei_stats_dict.copy(),
#                         # 'ucb_stats': ucb_stats_dict.copy(),
#                     }
#
#                 lf_data[lf_el] = n_reg_data
#                 # print(n_reg_data)
#             problem_data[problem_el.objective_function.name] = lf_data
#         model_type_data[model_type_el] = problem_data
#
#     return model_type_data, metadata_dict

def rqmc(pred_diff, samp_diff): # regression quality metric calculator
    L1E = np.abs(pred_diff).sum()  # L1 distance / prediction error
    L2E = np.sqrt((pred_diff ** 2).sum())  # L2 distance / prediction error

    L1D = np.abs(samp_diff).sum()  # L1 sample deviation (AAD / MAD)
    L2D = np.sqrt((samp_diff ** 2).sum())  # L2 sample deviation (Variance)

    L1E_div_L1D = L1E / L1D # "unexplained deviation" UD
    L2E_div_L2D = L2E / L2D # unexplained variance, (1 - R2) ** .5, FVU, UV

    L1E_div_L2D = L1E / (np.sqrt(len(pred_diff)) * L2D) # RMAE (RAAE)
    L2E_div_L1D = (np.sqrt(len(pred_diff)) * L2E) / L1D # RRMSE

    distance_vector = np.array([L1E, L2E])
    metric_matrix = np.array([[L1E_div_L1D, L1E_div_L2D], [L2E_div_L1D, L2E_div_L2D]])

    return distance_vector, metric_matrix


def reg_main(
        problem=None, model_type=None, lf=None, n_reg=None, n_reg_lf=None, scramble=False, random=False,
        n_inf=500, noise_fix=True, lf_jitter=1e-4, noise_type='b',
):
    total_training_time = 0
    if problem is None:
        problem = [
            MFProblem(
                objective_function=AugmentedTestFunction(
                    botorch_TestFunction(
                        function.Ackley(d=1), negate=False
                    )
                ).to(**tkwargs)
            ),
        ]

    if lf is None:
        lf = [0.5]

    if model_type is None:
        model_type = ['cokg']

    if n_reg is None:
        n_reg = [5]

    if n_reg_lf is None:
        n_reg_lf = [10]

    if scramble:
        n_DoE = 10
    else:
        n_DoE = 1

    # metadata_dict = {
    #     'problem': [p.objective_function.name for p in problem],
    #     'model_type': model_type,
    #     'lf': lf,
    #     'n_reg': n_reg,
    #     'n_reg_lf': n_reg_lf,
    #     'scramble': scramble,
    #     'noise_fix': noise_fix,
    #     'noise_type': noise_type,
    # }

    if not os.path.exists('reg_data'):
        os.mkdir('reg_data')

    # model_type_data = {}
    for model_type_el in model_type:
        # print()
        # print(model_type_el)

        problem_data = {}
        for problem_el in problem:
            # print()
            # print(problem_el.objective_function.name)

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim

            # lf_data = {}
            for lf_el in lf:
                # print()
                # print('lf =', lf_el)

                # n_reg_data = {}
                # mean_stats_dict = {}
                # std_stats_dict = {}
                for n_reg_el, n_reg_lf_el in zip(n_reg, n_reg_lf):
                    # print('n_reg =', n_reg_el)
                    # print('n_reg_lf_el =', n_reg_lf_el)
                    #
                    # n_DoE_mean_data = []
                    # n_DoE_std_data = []

                    for _ in range(n_DoE):
                        ####################
                        ### Pre-training ###
                        ####################

                        (train_x, train_y_high, train_obj, test_x_list, test_x_list_scaled, test_x_list_high,
                         scaler_y_high, exact_y, exact_y_low, train_y_low, scaler_y_low, ) = pretrainer(
                            problem_el, model_type_el, n_reg_el, n_reg_lf_el, lf_el, random, scramble, n_inf, bds, dim,
                        )

                        ################
                        ### Training ###
                        ################

                        start = time.time()
                        model = trainer(train_x, train_obj, problem_el, model_type_el, dim, noise_fix, lf_jitter, )
                        # if model_type_el in ['sogpr', 'cokg', 'mtask']:
                        #     print(model.state_dict())
                        #     if model_type_el in ['mtask']:
                        #         print(model.likelihood.noise_covar.raw_noise)

                        stop = time.time()
                        total_training_time += stop - start

                        ### NAME CONVENTION: dim, noise type, noise fix, objective name, LF parameter,
                        ### HF volume, LF volme, model type, iteration number, fidelity (cokg_dms only)
                        reg_problem_name = str(dim - 1) \
                                           + ',' + noise_type \
                                           + ',' + str(noise_fix) \
                                           + ',' + problem_el.objective_function.name \
                                           + ',' + str(lf_el) \
                                           + ',' + str(n_reg_el) + ',' + str(n_reg_lf_el) \
                                           # + ',' + model_type_el + ',' + str(_)
                        print()
                        print(reg_problem_name + ',' + model_type_el + ',' + str(_))
                        print('training time', stop - start)

                        if not os.path.exists('reg_data/' + reg_problem_name):
                            os.mkdir('reg_data/' + reg_problem_name)

                        if not os.path.exists('reg_data/' + reg_problem_name + '/' + model_type_el):
                            os.mkdir('reg_data/' + reg_problem_name + '/' + model_type_el)

                        if model_type_el != 'cokg_dms':
                            ### SAVING MODEL ###
                            torch.save(model.state_dict(), 'reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_))
                            # print(train_x)
                            torch.save(train_x, 'reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_x')
                            # print(train_obj)
                            torch.save(train_obj, 'reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_y')
                            # np.save('reg_data/' + reg_problem_name, model.state_dict())

                            # ### LOADING MODEL ###
                            # state_dict = torch.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_))
                            #
                            # # state_dict = np.load('reg_data/' + reg_problem_name + '.npy', allow_pickle=True)
                            #
                            # train_x_load = torch.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_x')
                            # # print(train_x_load)
                            # train_y_load = torch.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_y')
                            # # print(train_y_load)
                            #
                            # mll_load, model_load = problem_el.initialize_model(train_x_load, train_y_load, model_type=model_type_el, noise_fix=noise_fix)
                            # # print(model.state_dict())
                            # # print(model_load.state_dict())
                            # model_load.load_state_dict(state_dict)

                        else:
                            ### SAVING MODEL ###
                            train_x = np.array(train_x, dtype=object)
                            train_obj = np.array(train_obj, dtype=object)
                            np.save('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_x',
                                    train_x)
                            np.save('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_y',
                                    train_obj)
                            for cokg_dms_fid in range(2):
                                np.save('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + ',' + str(cokg_dms_fid),
                                        model.models[cokg_dms_fid].param_array)

                            # ### LOADING MODEL ###
                            # base_k = GPy.kern.RBF
                            # kernels_RL = [base_k(dim - 1) + GPy.kern.White(dim - 1), base_k(dim - 1)]
                            # train_x_load = np.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_x.npy', allow_pickle=True)
                            # train_y_load = np.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + '_train_y.npy', allow_pickle=True)
                            # model_load = GPy.models.multiGPRegression(
                            #     train_x_load,
                            #     train_y_load,
                            #     kernel=kernels_RL,
                            # )
                            # for cokg_dms_fid in range(2):
                            #     model_load.models[cokg_dms_fid].update_model(False)  # do not call the underlying expensive algebra on load
                            #     model_load.models[cokg_dms_fid].initialize_parameter()  # Initialize the parameters (connect the parameters up)
                            #     model_load.models[cokg_dms_fid][:] = np.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + ',' + str(cokg_dms_fid) + '.npy', allow_pickle=True)  # Load the parameters
                            #     model_load.models[cokg_dms_fid].update_model(True)  # Call the algebra only once

                        # #################################
                        # ### Post-training; prediction ###
                        # #################################
                        #
                        # (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
                        #  test_y_var_list_low,) = posttrainer(
                        #     model_load, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low
                        # )

                        # ########################
                        # ### Post-processing ####
                        # ########################
                        #
                        # # if model_type_el != 'cokg_dms':
                        # #     exact_y_scaled = model.outcome_transform(torch.tensor(exact_y))[0].detach().numpy()[0]
                        # # else:
                        # #     exact_y_scaled = scaler_y_high.transform(exact_y[:, None])
                        #
                        # # plt.figure(num='unscaled_' + problem_el.objective_function.name)
                        # # if model_type_el == 'sogpr':
                        # #     plt.plot(exact_y, '--', label='exact')
                        # # plt.plot(test_y_list_high, label=model_type_el)
                        # # plt.legend()
                        # # plt.figure(num='scaled_' + problem_el.objective_function.name)
                        # # if model_type_el == 'sogpr':
                        # #     plt.plot(exact_y_scaled, '--', label='exact_scaled')
                        # # plt.plot(test_y_list_high_scaled, label=model_type_el)
                        # # plt.legend()
                        #
                        # pred_diff = exact_y.flatten() - test_y_list_high.flatten()
                        # # pred_diff_scaled = exact_y_scaled.flatten() - test_y_list_high_scaled.flatten()
                        #
                        # samp_diff = exact_y - np.mean(exact_y)
                        # # samp_diff_scaled = exact_y_scaled - np.mean(exact_y_scaled)
                        #
                        # # print('unscaled', rqmc(pred_diff, samp_diff))
                        # # print('scaled', metric_calculator(pred_diff_scaled, samp_diff_scaled))
                        #
                        # # print('unscaled VAR', rqmc(np.sqrt(np.abs(test_y_var_list_high)), samp_diff))
                        # # print('scaled VAR', metric_calculator(np.sqrt(np.abs(test_y_var_list_high)), samp_diff_scaled))
                        #
                        # # RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
                        #
                        # print(rqmc(pred_diff, samp_diff)[1])
                        # # n_DoE_mean_data.append(rqmc(pred_diff, samp_diff))
                        # # n_DoE_std_data.append(rqmc(np.sqrt(np.abs(test_y_var_list_high)), samp_diff))
                        #
                        # vis2d = 0
                        # # if vis2d and model_type_el == 'mtask':
                        # if vis2d:
                        #     if dim - 1 == 2:
                        #         coord_mesh, _ = uniform_grid(bl=bds[0], tr=bds[1], n=[22, 22], mesh=True)
                        #         fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                        #                                num=problem_el.objective_function.name)
                        #         ax.plot_surface(coord_mesh[0], coord_mesh[1],
                        #                         test_y_list_high.reshape(coord_mesh[0].shape), cmap='viridis',
                        #                         linewidth=0, alpha=.5)
                        #         ax.scatter(train_x[:n_reg_el][:, 0], train_x[:n_reg_el][:, 1], train_y_high, c='r',
                        #                    s=50)
                        #         plt.tight_layout()
                        #
                        #     elif dim - 1 == 1:
                        #         plt.figure(num=problem_el.objective_function.name + '_' + model_type_el + str(_), figsize=(4, 4))
                        #         coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                        #         plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean')
                        #         plt.fill_between(coord_list.flatten(),
                        #                          (test_y_list_high - 2 * np.sqrt(
                        #                              np.abs(test_y_var_list_high))).flatten(),
                        #                          (test_y_list_high + 2 * np.sqrt(
                        #                              np.abs(test_y_var_list_high))).flatten(),
                        #                          alpha=.25, color='r', label='Predictive HF confidence interval')
                        #         plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
                        #         # if model_type_el != 'cokg_dms':
                        #         #     plt.scatter(train_x[:n_reg_el][:, 0], train_y_high, c='r', )
                        #         plt.legend()
                        #
                        #         show_low = 0
                        #         if model_type_el in ['cokg', 'cokg_dms', 'mtask'] and show_low:
                        #             # plt.figure(num=problem_el.objective_function.name + '_lf', figsize=(4, 4))
                        #             plt.plot(coord_list, test_y_list_low, 'b--')
                        #             plt.fill_between(coord_list.flatten(),
                        #                              (test_y_list_low - 2 * np.sqrt(
                        #                                  np.abs(test_y_var_list_low))).flatten(),
                        #                              (test_y_list_low + 2 * np.sqrt(
                        #                                  np.abs(test_y_var_list_low))).flatten(),
                        #                              alpha=.25, color='b')
                        #             plt.plot(coord_list, exact_y_low, 'b', linewidth=.5)
                        #             # if model_type_el != 'cokg_dms':
                        #             #     plt.scatter(train_x[n_reg_el:][:, 0], train_y_low, c='b', )
                        #
                        #         c = .1
                        #         plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                        #                   (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
                        #
                        #         plt.grid()
                        #         plt.tight_layout()

        #             mean_stats_dict['median'] = {
        #                 'distances': np.median([d[0] for d in n_DoE_mean_data], axis=0),
        #                 'metrics': np.median([d[1] for d in n_DoE_mean_data], axis=0),
        #             }
        #
        #             mean_stats_dict['quantiles'] = {
        #                 'distances': [np.quantile([d[0] for d in n_DoE_mean_data], q=.25, axis=0),
        #                               np.quantile([d[0] for d in n_DoE_mean_data], q=.75, axis=0)],
        #                 'metrics': [np.quantile([d[1] for d in n_DoE_mean_data], q=.25, axis=0),
        #                             np.quantile([d[1] for d in n_DoE_mean_data], q=.75, axis=0)],
        #             }
        #
        #             std_stats_dict['median'] = {
        #                 'distances': np.median([d[0] for d in n_DoE_std_data], axis=0),
        #                 'metrics': np.median([d[1] for d in n_DoE_std_data], axis=0),
        #             }
        #
        #             std_stats_dict['quantiles'] = {
        #                 'distances': [np.quantile([d[0] for d in n_DoE_std_data], q=.25, axis=0),
        #                               np.quantile([d[0] for d in n_DoE_std_data], q=.75, axis=0)],
        #                 'metrics': [np.quantile([d[1] for d in n_DoE_std_data], q=.25, axis=0),
        #                             np.quantile([d[1] for d in n_DoE_std_data], q=.75, axis=0)],
        #             }
        #
        #             n_reg_data[(n_reg_el, n_reg_lf_el)] = {
        #                 'mean_stats': mean_stats_dict.copy(),
        #                 'std_stats': std_stats_dict.copy(),
        #             }
        #
        #         lf_data[lf_el] = n_reg_data
        #     problem_data[problem_el.objective_function.name] = lf_data
        # model_type_data[model_type_el] = problem_data
        print()
        print('total training time', total_training_time)
    return # model_type_data, metadata_dict


def scale_to_unit(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (el[d] - bds[0][d]) / (bds[1][d] - bds[0][d])
        el_c += 1
    return res


# def bo_main_old(problem=None, model_type=None, lf=None, n_reg_init=None, scramble=True,
#                 n_inf=500, random=False, noise_fix=True, n_reg_lf_init=None, budget=None, ):
#     if problem is None:
#         problem = [
#             MFProblem(
#                 objective_function=AugmentedTestFunction(
#                     botorch_TestFunction(
#                         function.AlpineN2(d=1), negate=False
#                     )
#                 ).to(**tkwargs)
#             ),
#         ]
#
#     if lf is None:
#         lf = [0.5]
#
#     if model_type is None:
#         model_type = ['cokg']
#
#     if n_reg_init is None:
#         n_reg_init = [15]
#
#     if n_reg_lf_init is None:
#         n_reg_lf_init = [30]
#
#     metadata_dict = {
#         'problem': [p.objective_function.name for p in problem],
#         'model_type': model_type,
#         'lf': lf,
#         'n_reg_init': n_reg_init,
#         'n_reg_lf_init': n_reg_lf_init,
#         'scramble': scramble,
#         'noise_fix': noise_fix,
#         'budget': budget,
#         'noise_type': [p.objective_function.noise_type for p in problem][0],
#         'dim': [p.objective_function.dim for p in problem][0] - 1,
#         'cost_ratio': [p.cost_ratio for p in problem][0],
#     }
#
#     model_type_data = {}
#     for model_type_el in model_type:
#         print()
#         print(model_type_el)
#
#         problem_data = {}
#         for problem_el in problem:
#             print()
#             print(problem_el.objective_function.name)
#
#             if vis:
#                 plt.figure(num=problem_el.objective_function.name)
#             pm_one = 2 * (.5 - problem_el.objective_function.negate)
#
#             bds = problem_el.bounds
#             dim = problem_el.objective_function.dim
#             # xmin, ymin = problem_el.objective_function.opt(d=dim - 1)
#
#             lf_data = {}
#             for lf_el in lf:
#                 print()
#                 print('lf =', lf_el)
#
#                 n_reg_init_data = {}
#                 for n_reg_init_el, n_reg_lf_init_el in zip(n_reg_init, n_reg_lf_init):
#                     print()
#                     print('n_reg =', n_reg_init_el)
#                     print('n_reg_lf_el =', n_reg_lf_init_el)
#
#                     problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)
#
#                     (train_x, train_y_high, train_obj,
#                      test_x_list, test_x_list_scaled, test_x_list_high,
#                      scaler_y_high, exact_y, exact_y_low, train_y_low, scaler_y_low, ) = pretrainer(
#                         problem_el, model_type_el, n_reg_init_el, n_reg_lf_init_el, lf_el, random,
#                         scramble, n_inf, bds, dim,
#                     )
#
#                     train_x_init, train_y_init = train_x, train_obj
#
#                     train_x_high = torch.tensor([])
#                     train_obj_high = torch.tensor([])
#
#                     cumulative_cost = 0.0  # change into TOTAL cost (+ initial DoE cost)
#
#                     opt_data = {}
#                     _ = 0
#                     RAAEs, RMSTDs = [], []
#
#                     mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
#                     if noise_fix:
#                         cons = Interval(1e-4, 1e-4 + 1e-10)
#                         model.likelihood.noise_covar.register_constraint("raw_noise", cons)
#                     fit_gpytorch_model(mll)
#
#                     (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
#                      test_y_var_list_low,) = posttrainer(
#                         model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low,
#                     )
#
#                     RAAE = np.mean(np.abs(exact_y.reshape(np.shape(test_y_list_high)) - test_y_list_high)) / np.std(
#                         exact_y)
#                     RAAEs.append(RAAE)
#
#                     RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
#                     RMSTDs.append(RMSTD)
#
#                     opt_cost_history = [cumulative_cost]
#
#                     while cumulative_cost < budget:
#                         if _ % 5 == 0: print('iteration', _, ',', float(100 * cumulative_cost / budget), '% of budget')
#                         mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
#                         if noise_fix:
#                             cons = Interval(1e-4, 1e-4 + 1e-10)
#                             model.likelihood.noise_covar.register_constraint("raw_noise", cons)
#                         fit_gpytorch_model(mll)
#
#                         (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
#                          test_y_var_list_low) = posttrainer(
#                             model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low,
#                         )
#
#                         RAAE = np.mean(np.abs(exact_y.reshape(np.shape(test_y_list_high)) - test_y_list_high)) / np.std(
#                             exact_y)
#                         RAAEs.append(RAAE)
#
#                         RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
#                         RMSTDs.append(RMSTD)
#
#                         mfacq = problem_el.get_mfacq(model)
#                         new_x, new_obj, cost = problem_el.optimize_mfacq_and_get_observation(mfacq)
#
#                         if model_type_el != 'sogpr':
#                             train_x = torch.cat([train_x, new_x])
#                             train_obj = torch.cat([train_obj, new_obj])
#                             cumulative_cost += float(cost)
#
#                             if new_x[0][-1] == 1:
#                                 train_x_high = torch.cat([train_x_high, new_x])
#                                 train_obj_high = torch.cat([train_obj_high, new_obj])
#                         else:
#                             new_x = torch.cat([new_x, torch.tensor([1])])[:, None].T
#                             train_x = torch.cat([train_x, new_x])
#                             train_obj = torch.cat([train_obj, new_obj])
#                             cumulative_cost += 1
#
#                             train_x_high = torch.cat([train_x_high, new_x])
#                             train_obj_high = torch.cat([train_obj_high, new_obj])
#
#                         # if vis:
#                         #     size = 2 * (_ + 1)
#                         #     # plt.figure(model_type_el)
#                         #     if model_type_el is not 'sogpr':
#                         #         if new_x[0][-1] == 1:
#                         #             print('Sample at highest fidelity!')
#                         #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
#                         #                         color='g')
#                         #         else:
#                         #             # continue
#                         #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
#                         #                         color='b')
#                         #     else:
#                         #         plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
#                         #                     color='g')
#                         #     if cumulative_cost + cost > budget:
#                         #         plt.plot(test_x_list, pm_one * test_y_list_high, alpha=.1,
#                         #                  color='r')
#                         #         plt.fill_between(test_x_list.flatten(),
#                         #                          (pm_one * test_y_list_high - 2 * np.sqrt(
#                         #                              np.abs(test_y_var_list_high))).flatten(),
#                         #                          (pm_one * test_y_list_high + 2 * np.sqrt(
#                         #                              np.abs(test_y_var_list_high))).flatten(),
#                         #                          color='k', alpha=.05 #* (_ + 1) / n_iter_el
#                         #                          )
#                         #         if new_x[0][-1] == 1:
#                         #             print('Sample at highest fidelity!')
#                         #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
#                         #                         color='orange')
#                         #     plt.plot(test_x_list, pm_one * exact_y, 'k--')
#                         opt_cost_history.append(cumulative_cost)
#                         _ += 1
#
#                     # print(problem_el.get_recommendation(model))
#
#                     # print(train_x_init)
#
#                     # if len(train_obj_high) > 0:
#                     #     # print('Initial DoE EXCLUDED in calculating cumulative minimum')
#                     #     x_best = train_x_high[torch.argmax(train_obj_high)]
#                     #     y_best = -train_obj_high[torch.argmax(train_obj_high)]
#                     # else:
#                     #     # print('Initial DoE INCLUDED in calculating cumulative minimum')
#                     #     x_best = train_x[torch.argmax(train_obj)]
#                     #     y_best = -train_obj[torch.argmax(train_obj)]
#                     #
#                     # opt_data_vol = _ - 1
#                     #
#                     # x_err_norm = np.linalg.norm(x_best[:-1] - xmin) / np.linalg.norm(problem_el.bounds[1] - problem_el.bounds[0])
#                     # y_err_norm = np.linalg.norm(y_best - ymin) / (np.amax(-exact_y) - ymin)
#                     #
#                     # x_hist_norm = train_x[:, :-1]
#                     # y_hist_norm = np.linalg.norm(-train_obj - ymin, axis=1) / (np.amax(-exact_y) - ymin)
#
#                     # n_init_tot = n_reg_init_el + (model_type_el != 'sogpr') * n_reg_lf_init_el
#                     #
#                     # print(len(train_obj))
#                     #
#                     # hist_indices_high = [i for i, x in enumerate(train_x[n_init_tot:]) if x[-1] == 1]
#                     # y_hist_high = torch.tensor([train_obj[i] for i in hist_indices_high])
#                     # opt_cost_history_high = torch.tensor([opt_cost_history[i + 1] for i in hist_indices_high])
#                     #
#                     # vis_opt = True
#                     # if vis_opt:
#                     #     print(opt_cost_history_high, -train_obj_high, -y_hist_high)
#                     #     plt.figure(num=problem_el.objective_function.name)
#                     #     plt.plot(opt_cost_history_high, np.minimum.accumulate(-train_obj_high))
#                     #     plt.scatter(opt_cost_history_high, -y_hist_high)
#
#                     opt_data['x_hist'] = train_x.detach().numpy()
#                     # print(problem_el.objective_function.negate)
#                     opt_data['y_hist'] = pm_one * train_obj
#                     opt_data['RAAE'] = torch.tensor(RAAEs).to(**tkwargs)[:, None]
#                     opt_data['RMSTD'] = torch.tensor(RMSTDs).to(**tkwargs)[:, None]
#                     opt_data['opt_cost_history'] = opt_cost_history
#                     # opt_data['y_hist_norm'] = np.hstack((y_hist_norm[:, None], train_x[:, -1].detach().numpy()[:, None]))
#
#                     n_reg_init_data[(n_reg_init_el, n_reg_lf_init_el)] = opt_data
#
#                 lf_data[lf_el] = n_reg_init_data
#
#             problem_data[problem_el.objective_function.name] = lf_data
#
#         model_type_data[model_type_el] = problem_data
#
#     return model_type_data, metadata_dict


def bo_main(problem=None, model_type=None, lf=None, n_reg_init=None, scramble=True,
            n_inf=500, random=False, noise_fix=True, n_reg_lf_init=None, budget=None, ):
    if problem is None:
        problem = [
            MFProblem(
                objective_function=AugmentedTestFunction(
                    botorch_TestFunction(
                        function.AlpineN2(d=1), negate=False
                    )
                ).to(**tkwargs)
            ),
        ]

    if lf is None:
        lf = [0.5]

    if model_type is None:
        model_type = ['cokg']

    if n_reg_init is None:
        n_reg_init = [15]

    if n_reg_lf_init is None:
        n_reg_lf_init = [30]

    metadata_dict = {
        'problem': [p.objective_function.name for p in problem],
        'model_type': model_type,
        'lf': lf,
        'n_reg_init': n_reg_init,
        'n_reg_lf_init': n_reg_lf_init,
        'scramble': scramble,
        'noise_fix': noise_fix,
        'budget': budget,
        'noise_type': [p.objective_function.noise_type for p in problem][0],
        'dim': [p.objective_function.dim for p in problem][0] - 1,
        'cost_ratio': [p.cost_ratio for p in problem][0],
    }

    model_type_data = {}
    for model_type_el in model_type:
        print()
        print(model_type_el)

        problem_data = {}
        for problem_el in problem:
            print()
            print(problem_el.objective_function.name)

            if vis:
                plt.figure(num=problem_el.objective_function.name)
            pm_one = 2 * (.5 - problem_el.objective_function.negate)

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim

            lf_data = {}
            for lf_el in lf:
                print()
                print('lf =', lf_el)

                n_reg_init_data = {}
                for n_reg_init_el, n_reg_lf_init_el in zip(n_reg_init, n_reg_lf_init):
                    print()
                    print('n_reg =', n_reg_init_el)
                    print('n_reg_lf_el =', n_reg_lf_init_el)

                    problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)

                    (train_x, train_y_high, train_obj,
                     test_x_list, test_x_list_scaled, test_x_list_high,
                     scaler_y_high, exact_y, exact_y_low, train_y_low, scaler_y_low, ) = pretrainer(
                        problem_el, model_type_el, n_reg_init_el, n_reg_lf_init_el, lf_el, random,
                        scramble, n_inf, bds, dim,
                    )

                    train_x_init, train_y_init = train_x, train_obj

                    cumulative_cost = 0.0  # change into TOTAL cost (+ initial DoE cost)

                    opt_data = {}
                    iteration = 0
                    RAAEs, RMSTDs = [], []

                    mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
                    if noise_fix:
                        cons = Interval(1e-4, 1e-4 + 1e-10)
                        model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                    fit_gpytorch_model(mll)

                    (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
                     test_y_var_list_low,) = posttrainer(
                        model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low,
                    )

                    RAAE = np.mean(np.abs(exact_y.reshape(np.shape(test_y_list_high)) - test_y_list_high)) / np.std(
                        exact_y)
                    RAAEs.append(RAAE)

                    RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
                    RMSTDs.append(RMSTD)

                    opt_cost_history = [cumulative_cost]

                    while cumulative_cost < budget:
                        if iteration % 5 == 0: print('iteration', iteration, ',', float(100 * cumulative_cost / budget), '% of budget')
                        mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
                        if noise_fix:
                            cons = Interval(1e-4, 1e-4 + 1e-10)
                            model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                        fit_gpytorch_model(mll)

                        (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
                         test_y_var_list_low) = posttrainer(
                            model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low,
                        )

                        RAAE = np.mean(np.abs(exact_y.reshape(np.shape(test_y_list_high)) - test_y_list_high)) / np.std(
                            exact_y)
                        RAAEs.append(RAAE)

                        RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
                        RMSTDs.append(RMSTD)

                        mfacq = problem_el.get_mfacq(model)
                        new_x, new_obj, cost = problem_el.optimize_mfacq_and_get_observation(mfacq)

                        if model_type_el != 'sogpr':
                            train_x = torch.cat([train_x, new_x])
                            train_obj = torch.cat([train_obj, new_obj])
                            cumulative_cost += float(cost)

                        else:
                            new_x = torch.cat([new_x, torch.tensor([1])])[:, None].T
                            train_x = torch.cat([train_x, new_x])
                            train_obj = torch.cat([train_obj, new_obj])
                            cumulative_cost += 1

                        opt_cost_history.append(cumulative_cost)
                        iteration += 1

                    opt_data['x_hist'] = train_x.detach().numpy()
                    opt_data['y_hist'] = pm_one * train_obj
                    # opt_data['RAAE'] = torch.tensor(RAAEs).to(**tkwargs)[:, None]
                    # opt_data['RMSTD'] = torch.tensor(RMSTDs).to(**tkwargs)[:, None]
                    opt_data['opt_cost_history'] = opt_cost_history

                    n_reg_init_data[(n_reg_init_el, n_reg_lf_init_el)] = opt_data

                lf_data[lf_el] = n_reg_init_data

            problem_data[problem_el.objective_function.name] = lf_data

        model_type_data[model_type_el] = problem_data

    return model_type_data, metadata_dict
