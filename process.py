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

                        # ################
                        # ### Training ###
                        # ################
                        #
                        # start = time.time()
                        # model = trainer(train_x, train_obj, problem_el, model_type_el, dim, noise_fix, lf_jitter, )
                        # stop = time.time()
                        # total_training_time += stop - start

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
                        # print('training time', stop - start)

                        if not os.path.exists('reg_data/' + reg_problem_name):
                            os.mkdir('reg_data/' + reg_problem_name)

                        if not os.path.exists('reg_data/' + reg_problem_name + '/' + model_type_el):
                            os.mkdir('reg_data/' + reg_problem_name + '/' + model_type_el)

                        if model_type_el != 'cokg_dms':
                            # ### SAVING MODEL ###
                            # torch.save(model.state_dict(), 'reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_))
                            # # np.save('reg_data/' + reg_problem_name, model.state_dict())

                            ### LOADING MODEL ###
                            state_dict = torch.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_))
                            # state_dict = np.load('reg_data/' + reg_problem_name + '.npy', allow_pickle=True)
                            mll_load, model_load = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el, noise_fix=noise_fix)
                            model_load.load_state_dict(state_dict)

                        else:
                            # ### SAVING MODEL ###
                            # for cokg_dms_fid in range(2):
                            #     np.save('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + ',' + str(cokg_dms_fid),
                            #             model.models[cokg_dms_fid].param_array)

                            ### LOADING MODEL ###
                            base_k = GPy.kern.RBF
                            kernels_RL = [base_k(dim - 1) + GPy.kern.White(dim - 1), base_k(dim - 1)]
                            model_load = GPy.models.multiGPRegression(
                                train_x,
                                train_obj,
                                kernel=kernels_RL,
                            )
                            for cokg_dms_fid in range(2):
                                model_load.models[cokg_dms_fid].update_model(False)  # do not call the underlying expensive algebra on load
                                model_load.models[cokg_dms_fid].initialize_parameter()  # Initialize the parameters (connect the parameters up)
                                model_load.models[cokg_dms_fid][:] = np.load('reg_data/' + reg_problem_name + '/' + model_type_el + '/' + str(_) + ',' + str(cokg_dms_fid) + '.npy', allow_pickle=True)  # Load the parameters
                                model_load.models[cokg_dms_fid].update_model(True)  # Call the algebra only once

                        #################################
                        ### Post-training; prediction ###
                        #################################

                        (test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low,
                         test_y_var_list_low,) = posttrainer(
                            model_load, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, scaler_y_low
                        )

                        ########################
                        ### Post-processing ####
                        ########################

                        # if model_type_el != 'cokg_dms':
                        #     exact_y_scaled = model.outcome_transform(torch.tensor(exact_y))[0].detach().numpy()[0]
                        # else:
                        #     exact_y_scaled = scaler_y_high.transform(exact_y[:, None])

                        # plt.figure(num='unscaled_' + problem_el.objective_function.name)
                        # if model_type_el == 'sogpr':
                        #     plt.plot(exact_y, '--', label='exact')
                        # plt.plot(test_y_list_high, label=model_type_el)
                        # plt.legend()
                        # plt.figure(num='scaled_' + problem_el.objective_function.name)
                        # if model_type_el == 'sogpr':
                        #     plt.plot(exact_y_scaled, '--', label='exact_scaled')
                        # plt.plot(test_y_list_high_scaled, label=model_type_el)
                        # plt.legend()

                        pred_diff = exact_y.flatten() - test_y_list_high.flatten()
                        # pred_diff_scaled = exact_y_scaled.flatten() - test_y_list_high_scaled.flatten()

                        samp_diff = exact_y - np.mean(exact_y)
                        # samp_diff_scaled = exact_y_scaled - np.mean(exact_y_scaled)

                        # print('unscaled', rqmc(pred_diff, samp_diff))
                        # print('scaled', metric_calculator(pred_diff_scaled, samp_diff_scaled))

                        # print('unscaled VAR', rqmc(np.sqrt(np.abs(test_y_var_list_high)), samp_diff))
                        # print('scaled VAR', metric_calculator(np.sqrt(np.abs(test_y_var_list_high)), samp_diff_scaled))

                        # RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))

                        print(rqmc(pred_diff, samp_diff)[1])
                        # n_DoE_mean_data.append(rqmc(pred_diff, samp_diff))
                        # n_DoE_std_data.append(rqmc(np.sqrt(np.abs(test_y_var_list_high)), samp_diff))

                        vis2d = 0
                        if vis2d and model_type_el == 'mtask':
                            if dim - 1 == 2:
                                coord_mesh, _ = uniform_grid(bl=bds[0], tr=bds[1], n=[22, 22], mesh=True)
                                fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                                       num=problem_el.objective_function.name)
                                ax.plot_surface(coord_mesh[0], coord_mesh[1],
                                                test_y_list_high.reshape(coord_mesh[0].shape), cmap='viridis',
                                                linewidth=0, alpha=.5)
                                ax.scatter(train_x[:n_reg_el][:, 0], train_x[:n_reg_el][:, 1], train_y_high, c='r',
                                           s=50)
                                plt.tight_layout()

                            elif dim - 1 == 1:
                                plt.figure(num=problem_el.objective_function.name + '_' + model_type_el + str(_), figsize=(4, 4))
                                coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                plt.plot(coord_list, test_y_list_high, 'r--')
                                plt.fill_between(coord_list.flatten(),
                                                 (test_y_list_high - 2 * np.sqrt(
                                                     np.abs(test_y_var_list_high))).flatten(),
                                                 (test_y_list_high + 2 * np.sqrt(
                                                     np.abs(test_y_var_list_high))).flatten(),
                                                 alpha=.25, color='r')
                                plt.plot(coord_list, exact_y, 'r', linewidth=.5)
                                if model_type_el != 'cokg_dms':
                                    plt.scatter(train_x[:n_reg_el][:, 0], train_y_high, c='r', )

                                show_low = 0
                                if model_type_el in ['cokg', 'cokg_dms', 'mtask'] and show_low:
                                    # plt.figure(num=problem_el.objective_function.name + '_lf', figsize=(4, 4))
                                    plt.plot(coord_list, test_y_list_low, 'b--')
                                    plt.fill_between(coord_list.flatten(),
                                                     (test_y_list_low - 2 * np.sqrt(
                                                         np.abs(test_y_var_list_low))).flatten(),
                                                     (test_y_list_low + 2 * np.sqrt(
                                                         np.abs(test_y_var_list_low))).flatten(),
                                                     alpha=.25, color='b')
                                    plt.plot(coord_list, exact_y_low, 'b', linewidth=.5)
                                    if model_type_el != 'cokg_dms':
                                        plt.scatter(train_x[n_reg_el:][:, 0], train_y_low, c='b', )

                                c = .1
                                plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                                          (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])

                                plt.grid()
                                plt.tight_layout()

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