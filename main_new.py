import pickle
import sys

import pandas as pd

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
from pipeline_new import pretrainer, trainer, posttrainer

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


def scale_to_unit(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (el[d] - bds[0][d]) / (bds[1][d] - bds[0][d])
        el_c += 1
    return res


def rqmc(pred_diff, samp_diff):  # regression quality metric calculator
    L1E = np.abs(pred_diff).sum()  # L1 distance / prediction error
    L2E = np.sqrt((pred_diff ** 2).sum())  # L2 distance / prediction error

    L1D = np.abs(samp_diff).sum()  # L1 sample deviation (AAD / MAD)
    L2D = np.sqrt((samp_diff ** 2).sum())  # L2 sample deviation (standard deviation)

    L1E_div_L1D = L1E / L1D  # "unexplained deviation" UD
    L2E_div_L2D = L2E / L2D  # unexplained variance, (1 - R2) ** .5, FVU, UV

    L1E_div_L2D = L1E / (np.sqrt(len(pred_diff)) * L2D)  # RMAE (RAAE)
    L2E_div_L1D = (np.sqrt(len(pred_diff)) * L2E) / L1D  # RRMSE

    metric_matrix = L1E, L2E, L1E_div_L1D, L1E_div_L2D, L2E_div_L1D, L2E_div_L2D

    return metric_matrix


def reg_main(
        problem=None, model_type=None, lf=None, n_reg=None, n_reg_lf=None, scramble=False, random=False,
        n_inf=500, noise_fix=True, lf_jitter=1e-4, noise_type='b', optimize=True,
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

    for model_type_el in model_type:

        for problem_el in problem:

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim

            for lf_el in lf:

                for n_reg_el, n_reg_lf_el in zip(n_reg, n_reg_lf):

                    metrics = []

                    ### NAME CONVENTION: dim, noise type, noise fix, objective name, LF parameter,
                    ### HF volume, LF volme, model type, iteration number, fidelity (cokg_dms only)
                    reg_problem_name = str(dim - 1) \
                                       + ',' + noise_type \
                                       + ',' + str(noise_fix) \
                                       + ',' + problem_el.objective_function.name \
                                       + ',' + str(lf_el) \
                                       + ',' + str(n_reg_el) + ',' + str(n_reg_lf_el)

                    problem_path = 'reg_data/' + reg_problem_name

                    model_path = problem_path + '/' + model_type_el

                    for DoE_no in range(n_DoE):

                        ####################
                        ### Pre-training ###
                        ####################

                        train_x, train_obj, = pretrainer(
                            problem_el, model_type_el, n_reg_el, n_reg_lf_el, lf_el, random, scramble,
                        )

                        if not optimize:
                            if model_type_el == 'cokg_dms':
                                train_x = np.load(
                                    'reg_data/' + reg_problem_name + '/' + model_type_el + '/' +
                                    str(DoE_no) + '_train_x.npy', allow_pickle=True)
                                train_obj = np.load(
                                    'reg_data/' + reg_problem_name + '/' + model_type_el + '/' +
                                    str(DoE_no) + '_train_y.npy', allow_pickle=True)
                            else:
                                train_x = torch.load(
                                    'reg_data/' + reg_problem_name + '/' + model_type_el + '/' +
                                    str(DoE_no) + '_train_x')
                                train_obj = torch.load(
                                    'reg_data/' + reg_problem_name + '/' + model_type_el + '/' +
                                    str(DoE_no) + '_train_y')

                        ################
                        ### Training ###
                        ################

                        start = time.time()

                        model, scaler_y_high = trainer(
                            train_x, train_obj, problem_el, model_type_el, dim, bds, noise_fix, lf_jitter, n_reg_el,
                            optimize,
                        )

                        stop = time.time()
                        total_training_time += stop - start

                        exp_path = model_path + '/' + str(DoE_no)

                        if optimize:

                            ##############
                            ### Saving ###
                            ##############

                            print()
                            print(reg_problem_name + ',' + model_type_el + ',' + str(DoE_no))
                            print('training time', stop - start)

                            if not os.path.exists(problem_path): os.mkdir(problem_path)

                            if not os.path.exists(model_path): os.mkdir(model_path)

                            if model_type_el != 'cokg_dms':
                                ### SAVING MODEL ###
                                torch.save(model.state_dict(), exp_path)
                                torch.save(train_x, exp_path + '_train_x')
                                torch.save(train_obj, exp_path + '_train_y')

                            else:
                                ### SAVING MODEL ###
                                train_x = np.array(train_x, dtype=object)
                                train_obj = np.array(train_obj, dtype=object)
                                np.save(exp_path + '_train_x', train_x)
                                np.save(exp_path + '_train_y', train_obj)
                                for cokg_dms_fid in range(2):
                                    np.save(exp_path + ',' + str(cokg_dms_fid), model.models[cokg_dms_fid].param_array)

                        else:

                            ###############
                            ### Loading ###
                            ###############

                            if model_type_el != 'cokg_dms':
                                ### LOADING MODEL ###
                                state_dict = torch.load(exp_path)
                                model.load_state_dict(state_dict)

                            else:
                                ### LOADING MODEL ###
                                for cokg_dms_fid in range(2):
                                    model.models[cokg_dms_fid].update_model(
                                        False)  # do not call the underlying expensive algebra on load
                                    model.models[
                                        cokg_dms_fid].initialize_parameter()  # Initialize the parameters (connect the parameters up)
                                    model.models[cokg_dms_fid][:] = np.load(exp_path + ',' + str(cokg_dms_fid) + '.npy',
                                                                            allow_pickle=True)  # Load the parameters
                                    model.models[cokg_dms_fid].update_model(True)  # Call the algebra only once

                            #################################
                            ### Post-training; prediction ###
                            #################################

                            test_y_list_high, test_y_var_list_high, exact_y = posttrainer(
                                model, model_type_el, problem_el, bds, dim, n_inf, scaler_y_high
                            )

                            # plt.figure(num='hist')
                            # plt.hist((exact_y.flatten() - test_y_list_high.flatten()) / np.sqrt(np.abs(test_y_var_list_high.flatten())), ec='k', bins=50)

                            ########################
                            ### Post-processing ####
                            ########################

                            pred_diff = exact_y.flatten() - test_y_list_high.flatten()
                            # pred_diff_scaled = exact_y_scaled.flatten() - test_y_list_high_scaled.flatten()

                            samp_diff = exact_y - np.mean(exact_y)

                            metrics.append(rqmc(pred_diff, samp_diff))

                            # RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))
                            # print()
                            # print(rqmc(pred_diff, samp_diff)[1])
                            # n_DoE_mean_data.append(rqmc(pred_diff, samp_diff))
                            # n_DoE_std_data.append(rqmc(np.sqrt(np.abs(test_y_var_list_high)), samp_diff))

                            vis2d = 0
                            if vis2d:
                                if dim - 1 == 2:
                                    coord_mesh, _ = uniform_grid(bl=bds[0], tr=bds[1], n=[22, 22], mesh=True)
                                    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                                           num=problem_el.objective_function.name)
                                    ax.plot_surface(coord_mesh[0], coord_mesh[1],
                                                    test_y_list_high.reshape(coord_mesh[0].shape), cmap='viridis',
                                                    linewidth=0, alpha=.5)
                                    # ax.scatter(train_x[:n_reg_el][:, 0], train_x[:n_reg_el][:, 1], train_y_high, c='r',
                                    #            s=50)

                                elif dim - 1 == 1:
                                    plt.figure(num=problem_el.objective_function.name + 'DoE_no' + model_type_el,
                                               figsize=(4, 4))
                                    coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                    plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean',
                                             alpha=.2, )
                                    plt.fill_between(coord_list.flatten(),
                                                     (test_y_list_high - 2 * np.sqrt(
                                                         np.abs(test_y_var_list_high))).flatten(),
                                                     (test_y_list_high + 2 * np.sqrt(
                                                         np.abs(test_y_var_list_high))).flatten(),
                                                     alpha=.025, color='r', label='Predictive HF confidence interval')
                                    if DoE_no == 0:
                                        plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
                                        plt.grid()
                                    # if model_type_el != 'cokg_dms':
                                    #     plt.scatter(train_x[:n_reg_el][:, 0], train_y_high, c='r', )
                                    # plt.legend()

                                    # show_low = 0
                                    # if model_type_el in ['cokg', 'cokg_dms', 'mtask'] and show_low:
                                    #     # plt.figure(num=problem_el.objective_function.name + '_lf', figsize=(4, 4))
                                    #     plt.plot(coord_list, test_y_list_low, 'b--')
                                    #     plt.fill_between(coord_list.flatten(),
                                    #                      (test_y_list_low - 2 * np.sqrt(
                                    #                          np.abs(test_y_var_list_low))).flatten(),
                                    #                      (test_y_list_low + 2 * np.sqrt(
                                    #                          np.abs(test_y_var_list_low))).flatten(),
                                    #                      alpha=.25, color='b')
                                    #     plt.plot(coord_list, exact_y_low, 'b', linewidth=.5)
                                    #     # if model_type_el != 'cokg_dms':
                                    #     #     plt.scatter(train_x[n_reg_el:][:, 0], train_y_low, c='b', )

                                    c = .1
                                    plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                                              (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
                                plt.tight_layout()

                    if not optimize:
                        if not os.path.exists('reg_data/pp_data/'): os.mkdir('reg_data/pp_data/')

                        # print()
                        print('reg_data/pp_data/' + reg_problem_name + ',' + model_type_el + '.csv')
                        df = pd.DataFrame(
                            metrics,
                            columns=['L1E', 'L2E', 'L1E/L1D', 'L1E/L2D', 'L2E/L1D', 'L2E/L2D'],
                        )
                        # print(df)
                        df.to_csv('reg_data/pp_data/' + reg_problem_name + ',' + model_type_el + '.csv')
        if optimize:
            print()
            print('total training time', total_training_time)
    return


def bo_main(problem=None, model_type=None, lf=None, n_reg_init=None, scramble=True,
            n_inf=500, random=False, noise_fix=True, noise_type='b', n_reg_lf_init=None, max_budget=None,
            post_processing=False, acq_type='EI', iter_thresh=100, dev=False, opt_problem_name='exp_test'):
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

    if scramble:
        n_DoE = 10
    else:
        n_DoE = 1

    opt_medians_model_types = []

    for model_type_el in model_type:

        opt_medians = []

        for problem_el in problem:
            # print(problem_el.objective_function.name)

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim

            for lf_el in lf:

                for n_reg_init_el, n_reg_lf_init_el in zip(n_reg_init, n_reg_lf_init):

                    if not post_processing:

                        for DoE_no in range(n_DoE):
                            print(DoE_no)
                            problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)

                            train_x, train_obj = pretrainer(
                                problem_el,
                                model_type_el,
                                n_reg_init_el,
                                n_reg_lf_init_el,
                                lf_el,
                                random,
                                scramble,
                            )

                            if model_type_el != 'sogpr':
                                init_costs = n_reg_init_el * [1] + n_reg_lf_init_el * [1 / problem_el.cost_ratio]
                            else:
                                init_costs = n_reg_init_el * [1]

                            iteration = 0

                            mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
                            if noise_fix:
                                cons = Interval(1e-4, 1e-4 + 1e-10)
                                model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                            fit_gpytorch_model(mll)

                            test_y_list_high, test_y_var_list_high, exact_y, \
                            test_y_list_low, test_y_var_list_low, exact_y_low = posttrainer(
                                model,
                                model_type_el,
                                problem_el,
                                bds,
                                dim,
                                n_inf,
                                scaler_y_high=None,
                            )

                            vis_opt = 0
                            if vis_opt and dim - 1 == 1:
                                plt.figure(num=problem_el.objective_function.name + '_init_' + str(DoE_no))
                                coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                coord_list_tensor = torch.tensor(coord_list)
                                plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean', )
                                plt.fill_between(coord_list.flatten(),
                                                 (test_y_list_high - 2 * np.sqrt(
                                                     np.abs(test_y_var_list_high))).flatten(),
                                                 (test_y_list_high + 2 * np.sqrt(
                                                     np.abs(test_y_var_list_high))).flatten(),
                                                 alpha=.25, color='r', label='Predictive HF confidence interval')
                                plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
                                train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
                                train_obj_high = torch.stack(
                                    [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])
                                plt.scatter(train_x_high, train_obj_high, c='r')

                                if model_type_el != 'sogpr':
                                    train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                                    train_obj_low = torch.stack(
                                        [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])
                                    plt.scatter(train_x_low, train_obj_low, c='g')

                                c = .1
                                plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                                          (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
                                plt.tight_layout()

                            opt_cost_history = list(np.cumsum(init_costs))
                            rqmc_history = [
                                rqmc(exact_y.flatten() - test_y_list_high.flatten(), exact_y - np.mean(exact_y))]
                            cir_history = [2 * np.amax(np.sqrt(np.abs(test_y_var_list_high)))]
                            cumulative_cost = 0  # opt_cost_history[-1]
                            _, y_min = problem_el.objective_function.opt(d=dim - 1)

                            if problem_el.objective_function.name == 'AugmentedQing':
                                y_min = y_min[0]

                            train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
                            train_obj_high = torch.stack([y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

                            if model_type_el != 'sogpr':
                                train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                                train_obj_low = torch.stack([y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

                            train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
                            train_obj_high = torch.stack(
                                [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

                            if model_type_el != 'sogpr':
                                train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                                train_obj_low = torch.stack(
                                    [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

                            while cumulative_cost < max_budget and iteration < iter_thresh:
                                if iteration % 5 == 0: print('iteration', iteration, ',',
                                                             float(100 * cumulative_cost / max_budget),
                                                             '% of max. budget')

                                mfacq = problem_el.get_mfacq(model, y=train_obj_high, acq_type=acq_type)
                                new_x, new_obj, cost = problem_el.optimize_mfacq_and_get_observation(mfacq)
                                cost = 1 if cost is None else cost
                                cumulative_cost += float(cost)

                                if model_type_el != 'sogpr':
                                    train_x = torch.cat([train_x, new_x])
                                    train_obj = torch.cat([train_obj, new_obj])

                                else:
                                    new_x = torch.cat([new_x, torch.tensor([1])])[:, None].T
                                    train_x = torch.cat([train_x, new_x])
                                    train_obj = torch.cat([train_obj, new_obj])

                                opt_cost_history.append(cumulative_cost)

                                train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
                                train_obj_high = torch.stack(
                                    [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

                                if model_type_el != 'sogpr':
                                    train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                                    train_obj_low = torch.stack(
                                        [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

                                mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
                                if noise_fix:
                                    cons = Interval(1e-4, 1e-4 + 1e-10)
                                    model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                                fit_gpytorch_model(mll)

                                test_y_list_high, test_y_var_list_high, exact_y, \
                                test_y_list_low, test_y_var_list_low, exact_y_low = posttrainer(
                                    model,
                                    model_type_el,
                                    problem_el,
                                    bds,
                                    dim,
                                    n_inf,
                                    scaler_y_high=None,
                                )

                                cir_history.append(2 * np.amax(np.sqrt(np.abs(test_y_var_list_high))))

                                rqmc_history.append(
                                    rqmc(exact_y.flatten() - test_y_list_high.flatten(), exact_y - np.mean(exact_y))
                                )

                                if vis_opt and dim - 1 == 1:
                                    plt.figure(num=problem_el.objective_function.name + '_' + str(iteration))
                                    coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                    plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean')
                                    plt.fill_between(coord_list.flatten(),
                                                     (test_y_list_high - 2 * np.sqrt(
                                                         np.abs(test_y_var_list_high))).flatten(),
                                                     (test_y_list_high + 2 * np.sqrt(
                                                         np.abs(test_y_var_list_high))).flatten(),
                                                     alpha=.25, color='r', label='Predictive HF confidence interval')
                                    plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
                                    train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
                                    train_obj_high = torch.stack(
                                        [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])
                                    plt.scatter(train_x_high, train_obj_high, c='r')

                                    if model_type_el != 'sogpr':
                                        # print(test_y_list_low)
                                        plt.plot(coord_list, test_y_list_low, 'g--', label='Predictive LF mean')
                                        plt.fill_between(coord_list.flatten(),
                                                         (test_y_list_low - 2 * np.sqrt(
                                                             np.abs(test_y_var_list_high))).flatten(),
                                                         (test_y_list_low + 2 * np.sqrt(
                                                             np.abs(test_y_var_list_high))).flatten(),
                                                         alpha=.25, color='g',
                                                         label='Predictive LF confidence interval')
                                        plt.plot(coord_list, exact_y_low, 'g.', alpha=.2, linewidth=.5,
                                                 label='Exact LF objective')
                                        train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                                        train_obj_low = torch.stack(
                                            [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])
                                        plt.scatter(train_x_low, train_obj_low, c='g')

                                    col = 'r' if train_x[-1, -1] == 1 else 'g'
                                    plt.scatter(train_x[-1, 0], train_obj[-1], c=col, alpha=.5, s=150)
                                    c = .1
                                    # plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                                    #           (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
                                    plt.tight_layout()

                                    plt.figure(num='acq' + str(iteration))
                                    if model_type_el == 'sogpr':
                                        mfacq_eval = torch.stack([mfacq.forward(x[:, None]) for x in coord_list_tensor])
                                        plt.plot(coord_list, mfacq_eval.detach().numpy())
                                    else:
                                        mfacq_eval_high = torch.stack([mfacq.forward(
                                            torch.cat((x, torch.tensor([1])))[None, :]
                                        ) for x in coord_list_tensor])
                                        mfacq_eval_low = torch.stack([mfacq.forward(
                                            torch.cat((x, torch.tensor([lf_el])))[None, :]
                                        ) for x in coord_list_tensor])
                                        plt.plot(coord_list, mfacq_eval_high.detach().numpy(), label='high')
                                        plt.plot(coord_list, mfacq_eval_low.detach().numpy(), label='low')
                                        plt.legend()

                                iteration += 1

                            x_names = ['x_' + str(i) for i in range(dim - 1)] + ['fid']

                            if model_type_el != 'sogpr':
                                x_low_rec = problem_el.optimize_mfacq_and_get_observation(mfacq, final=0)[0]
                                x_low_rec[0][-1] = 1.0

                                x_high_rec, y_high_rec, _ \
                                    = problem_el.optimize_mfacq_and_get_observation(mfacq, final=1)

                                # print(problem_el.objective_function(x_low_rec), y_high_rec)
                                # print(x_low_rec, x_high_rec)

                                y_rec = np.minimum(
                                    problem_el.objective_function(x_low_rec),
                                    y_high_rec
                                )

                                if y_rec.flatten() == y_high_rec.flatten():
                                    x_rec = x_high_rec
                                else:
                                    x_rec = x_low_rec

                                df_rec = pd.DataFrame(
                                    np.hstack((x_rec, y_rec - y_min)),
                                    columns=x_names + ['y - y_min']
                                )

                            else:
                                x_rec = train_x_high[torch.argmin(train_obj_high)]
                                y_rec = torch.amin(train_obj_high)

                                df_rec = pd.DataFrame(
                                    np.hstack((x_rec, y_rec - y_min))[None, :],
                                    columns=x_names[:-1] + ['y - y_min']
                                )

                            df = pd.DataFrame(columns=['cost', 'y - y_min'] + x_names,
                                              data=np.hstack((np.array(opt_cost_history)[:, None],
                                                              train_obj.numpy() - y_min,
                                                              train_x.numpy(),)))
                            df2 = pd.DataFrame(
                                index=list(
                                    range(n_reg_init_el + (model_type_el != 'sogpr') * n_reg_lf_init_el - 1, len(df))),
                                columns=['L1E', 'L2E', 'L1E/L1D', 'L1E/L2D', 'L2E/L1D', 'L2E/L2D'],
                                data=np.array(rqmc_history)
                            )

                            df3 = pd.DataFrame(
                                index=list(
                                    range(n_reg_init_el + (model_type_el != 'sogpr') * n_reg_lf_init_el - 1, len(df))),
                                columns=['cir'],
                                data=np.array(cir_history)
                            )

                            df_res = pd.concat([df, df2, df3], axis=1)

                            # print(df_res)

                            if not os.path.exists('opt_data'):
                                os.mkdir('opt_data')

                            if not os.path.exists('opt_data_dev'):
                                os.mkdir('opt_data_dev')

                            ### NAME CONVENTION: dim, noise type, noise fix, LF parameter,
                            ### HF volume, LF volume
                            # opt_problem_name = str(dim - 1) + '_d' \
                            #                    + ',' + noise_type + '_nt' \
                            #                    + ',' + str(noise_fix) + '_nf' \
                            #                    + ',' + str(lf_el) + '_lf' \
                            #                    + ',' + str(n_reg_init_el) + '_nh,' + str(n_reg_lf_init_el) + '_nl' \
                            #                    + ',' + str(max_budget) + '_b' \
                            #                    + ',' + str(problem_el.cost_ratio) + '_cr'

                            if dev:
                                opt_problem_path = 'opt_data_dev/' + opt_problem_name
                            else:
                                opt_problem_path = 'opt_data/' + opt_problem_name

                            if not os.path.exists(opt_problem_path):
                                os.mkdir(opt_problem_path)

                            objective_path = opt_problem_path + '/' + problem_el.objective_function.name

                            if not os.path.exists(objective_path):
                                os.mkdir(objective_path)

                            model_problem_path = objective_path + '/' + model_type_el

                            if not os.path.exists(model_problem_path):
                                os.mkdir(model_problem_path)

                            DoE_no_path = model_problem_path + '/' + str(DoE_no)
                            print(DoE_no_path)

                            df_rec.to_csv(DoE_no_path + '_rec.csv')
                            df_res.to_csv(DoE_no_path + '.csv')

                    else:

                        hist_high_opt_mins = []

                        for DoE_no in range(n_DoE):
                            ### NAME CONVENTION: dim, noise type, noise fix, LF parameter,
                            ### HF volume, LF volume

                            # if model_type_el == 'stmf':
                            #     lf_el = 0.99
                            #     problem_el.cost_ratio = 55

                            # opt_problem_name = str(dim - 1) + '_d' \
                            #                    + ',' + noise_type + '_nt' \
                            #                    + ',' + str(noise_fix) + '_nf' \
                            #                    + ',' + str(lf_el) + '_lf' \
                            #                    + ',' + str(n_reg_init_el) + '_nh,' + str(n_reg_lf_init_el) + '_nl' \
                            #                    + ',' + str(max_budget) + '_b' \
                            #                    + ',' + str(problem_el.cost_ratio) + '_cr'

                            if dev:
                                opt_problem_path = 'opt_data_dev/' + opt_problem_name
                            else:
                                opt_problem_path = 'opt_data/' + opt_problem_name

                            objective_path = opt_problem_path + '/' + problem_el.objective_function.name

                            model_problem_path = objective_path + '/' + model_type_el

                            DoE_no_path = model_problem_path + '/' + str(DoE_no) + '.csv'

                            df = pd.read_csv(DoE_no_path)
                            print(df)
                            cost, hist = df['cost'], df['y - y_min']
                            high_opt_indices = [i for i, fid in enumerate(df['fid']) if fid == 1 and i >= n_reg_init_el]

                            cost_high_opt, hist_high_opt = \
                                np.array([cost[i] for i in high_opt_indices]), np.array(
                                    [hist[i] for i in high_opt_indices])

                            # print(np.minimum.accumulate(hist_high_opt))
                            if len(hist_high_opt) > 0:
                                hist_high_opt_mins.append(np.minimum.accumulate(hist_high_opt)[-1])

                            vis_opt_process = 1
                            if vis_opt_process:
                                plt.figure(num=problem_el.objective_function.name)

                                if model_type_el == 'sogpr':
                                    c = 'b'
                                else:
                                    c = 'g'

                                plt.plot(cost_high_opt, np.minimum.accumulate(hist_high_opt), color=c,
                                         label=model_type_el)
                                # plt.scatter(cost_high_opt, hist_high_opt, c=c)
                                plt.yscale('log')
                                plt.legend()

                        opt_median = np.median(hist_high_opt_mins)
                        opt_medians.append(opt_median)

        opt_medians_model_types.append(np.array(opt_medians))

    # print(opt_medians_model_types[0])
    # print(opt_medians_model_types[1])

    if post_processing:
        log_improvement = np.log10(
            np.maximum(opt_medians_model_types[0], 1e-9) / np.maximum(opt_medians_model_types[1], 1e-9))
        log_improvement.sort()
        log_improvement = log_improvement[~np.isnan(log_improvement)]
        # print(log_improvement)
        plt.figure(num='histo')
        plt.hist(log_improvement, ec='k', bins=30)
        print('lf', lf_el, 'cr', problem_el.cost_ratio, str(np.median(log_improvement)), str(np.mean(log_improvement)))
        plt.title(str(np.median(log_improvement)) + ', ' + str(np.mean(log_improvement)))

    return


def bo_main_unit(problem_el=None, model_type_el=None, lf_el=None, n_reg_init_el=None, scramble=True, cost_ratio=10,
                 n_inf=500, random=False, noise_fix=True, noise_type='b', n_reg_lf_init_el=None, max_budget=None, vis_opt=False,
                 post_processing=False, acq_type='EI', iter_thresh=100, dev=False, opt_problem_name='exp_test', n_DoE=0, exp_name='exp_0'):

    bds = problem_el.bounds
    dim = problem_el.objective_function.dim

    problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)
    problem_el.objective_function.noise_type = noise_type
    problem_el.cost_ratio = cost_ratio

    train_x, train_obj = pretrainer(
        problem_el,
        model_type_el,
        n_reg_init_el,
        n_reg_lf_init_el,
        lf_el,
        random,
        scramble,
    )

    if model_type_el != 'sogpr':
        init_costs = n_reg_init_el * [1] + n_reg_lf_init_el * [1 / problem_el.cost_ratio]
    else:
        init_costs = n_reg_init_el * [1]

    iteration = 0

    mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
    if noise_fix:
        cons = Interval(1e-4, 1e-4 + 1e-10)
        model.likelihood.noise_covar.register_constraint("raw_noise", cons)
    fit_gpytorch_model(mll)

    test_y_list_high, test_y_var_list_high, exact_y, \
    test_y_list_low, test_y_var_list_low, exact_y_low = posttrainer(
        model,
        model_type_el,
        problem_el,
        bds,
        dim,
        n_inf,
        scaler_y_high=None,
    )

    if vis_opt and dim - 1 == 1:
        plt.figure(num=problem_el.objective_function.name + '_init_' + str(n_DoE))
        coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
        coord_list_tensor = torch.tensor(coord_list)
        plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean', )
        plt.fill_between(coord_list.flatten(),
                         (test_y_list_high - 2 * np.sqrt(
                             np.abs(test_y_var_list_high))).flatten(),
                         (test_y_list_high + 2 * np.sqrt(
                             np.abs(test_y_var_list_high))).flatten(),
                         alpha=.25, color='r', label='Predictive HF confidence interval')
        plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
        train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
        train_obj_high = torch.stack(
            [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])
        plt.scatter(train_x_high, train_obj_high, c='r')

        if model_type_el != 'sogpr':
            train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
            train_obj_low = torch.stack(
                [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])
            plt.scatter(train_x_low, train_obj_low, c='g')

        c = .1
        plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
                  (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
        plt.tight_layout()

    opt_cost_history = list(np.cumsum(init_costs))
    rqmc_history = [
        rqmc(exact_y.flatten() - test_y_list_high.flatten(), exact_y - np.mean(exact_y))]
    cir_history = [2 * np.amax(np.sqrt(np.abs(test_y_var_list_high)))]
    cumulative_cost = 0  # opt_cost_history[-1]
    _, y_min = problem_el.objective_function.opt(d=dim - 1)

    if problem_el.objective_function.name == 'AugmentedQing':
        y_min = y_min[0]

    train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
    train_obj_high = torch.stack([y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

    if model_type_el != 'sogpr':
        train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
        train_obj_low = torch.stack([y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

    train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
    train_obj_high = torch.stack(
        [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

    if model_type_el != 'sogpr':
        train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
        train_obj_low = torch.stack(
            [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

    work_folder_name = 'opt_data'
    if dev:
        work_folder_name = 'opt_data_dev'

    if not os.path.exists(work_folder_name):
        os.mkdir(work_folder_name)

    if not os.path.exists(work_folder_name + '/' + exp_name):
        os.mkdir(work_folder_name + '/' + exp_name)

    opt_problem_path = work_folder_name + '/' + exp_name + '/' + opt_problem_name

    if not os.path.exists(opt_problem_path):
        os.mkdir(opt_problem_path)

    objective_path = opt_problem_path + '/' + problem_el.objective_function.name

    if not os.path.exists(objective_path):
        os.mkdir(objective_path)

    model_problem_path = objective_path + '/' + model_type_el

    if not os.path.exists(model_problem_path):
        os.mkdir(model_problem_path)

    DoE_no_path = model_problem_path + '/' + str(n_DoE)

    while cumulative_cost < max_budget and iteration < iter_thresh:
        if iteration % 5 == 0: print('iteration', iteration, ',',
                                     float(100 * cumulative_cost / max_budget),
                                     '% of max. budget')

        test_y_list_high, test_y_var_list_high, exact_y, \
        test_y_list_low, test_y_var_list_low, exact_y_low = posttrainer(
            model,
            model_type_el,
            problem_el,
            bds,
            dim,
            n_inf,
            scaler_y_high=None,
        )

        mfacq = problem_el.get_mfacq(
            model, y=train_obj_high, acq_type=acq_type,
            mean=[test_y_list_low, test_y_list_high], var=[test_y_var_list_low, test_y_var_list_high],
        )
        new_x, new_obj, cost = problem_el.optimize_mfacq_and_get_observation(mfacq)
        cost = 1 if cost is None else cost
        cumulative_cost += float(cost)

        if model_type_el != 'sogpr':
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

        else:
            new_x = torch.cat([new_x, torch.tensor([1])])[:, None].T
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

        opt_cost_history.append(cumulative_cost)

        train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
        train_obj_high = torch.stack(
            [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])

        if model_type_el != 'sogpr':
            train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
            train_obj_low = torch.stack(
                [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])

        mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
        if noise_fix:
            cons = Interval(1e-4, 1e-4 + 1e-10)
            model.likelihood.noise_covar.register_constraint("raw_noise", cons)
        fit_gpytorch_model(mll)

        cir_history.append(2 * np.amax(np.sqrt(np.abs(test_y_var_list_high))))

        rqmc_history.append(
            rqmc(exact_y.flatten() - test_y_list_high.flatten(), exact_y - np.mean(exact_y))
        )

        if vis_opt and dim - 1 == 1 and n_DoE == 0:

            if not os.path.exists(work_folder_name + '/' + exp_name + '/img'):
                os.mkdir(work_folder_name + '/' + exp_name + '/img')

            if not os.path.exists(work_folder_name + '/' + exp_name + '/img/' + problem_el.objective_function.name):
                os.mkdir(work_folder_name + '/' + exp_name + '/img/' + problem_el.objective_function.name)

            plt.figure(num=problem_el.objective_function.name + '_' + model_type_el + '_' + str(iteration))
            coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
            plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean')
            plt.fill_between(coord_list.flatten(),
                             (test_y_list_high - 2 * np.sqrt(
                                 np.abs(test_y_var_list_high))).flatten(),
                             (test_y_list_high + 2 * np.sqrt(
                                 np.abs(test_y_var_list_high))).flatten(),
                             alpha=.25, color='r', label='Predictive HF confidence interval')
            plt.plot(coord_list, exact_y, 'r', linewidth=.5, label='Exact HF objective')
            train_x_high = torch.stack([x[:-1] for x in train_x if x[-1] == 1])
            train_obj_high = torch.stack(
                [y for i, y in enumerate(train_obj) if train_x[i, -1] == 1])
            plt.scatter(train_x_high, train_obj_high, c='r')

            if model_type_el != 'sogpr':
                # print(test_y_list_low)
                plt.plot(coord_list, test_y_list_low, 'g--', label='Predictive LF mean')
                plt.fill_between(coord_list.flatten(),
                                 (test_y_list_low - 2 * np.sqrt(
                                     np.abs(test_y_var_list_high))).flatten(),
                                 (test_y_list_low + 2 * np.sqrt(
                                     np.abs(test_y_var_list_high))).flatten(),
                                 alpha=.25, color='g',
                                 label='Predictive LF confidence interval')
                plt.plot(coord_list, exact_y_low, 'g.', alpha=.2, linewidth=.5, label='Exact LF objective')
                train_x_low = torch.stack([x[:-1] for x in train_x if x[-1] != 1])
                train_obj_low = torch.stack(
                    [y for i, y in enumerate(train_obj) if train_x[i, -1] != 1])
                plt.scatter(train_x_low, train_obj_low, c='g')

            col = 'r' if train_x[-1, -1] == 1 else 'g'
            plt.scatter(train_x[-1, 0], train_obj[-1], c=col, alpha=.5, s=150)
            plt.title(problem_el.objective_function.name + '_' + str(iteration))
            # c = .1
            # plt.ylim([(1 + c) * np.amin(exact_y) - c * np.amax(exact_y),
            #           (1 + c) * np.amax(exact_y) - c * np.amin(exact_y)])
            plt.tight_layout()

            plt.savefig(work_folder_name + '/' + exp_name + '/img/' + problem_el.objective_function.name + '/iter' + '_' + str(iteration) + '.png')

            plt.figure(num='acq' + '_' + model_type_el + str(iteration))
            if model_type_el == 'sogpr':
                mfacq_eval = torch.stack([mfacq.forward(x[:, None]) for x in coord_list_tensor])
                plt.plot(coord_list, mfacq_eval.detach().numpy())
            else:
                mfacq_eval_high = torch.stack([mfacq.forward(
                    torch.cat((x, torch.tensor([1])))[None, :]
                ) for x in coord_list_tensor])
                mfacq_eval_low = torch.stack([mfacq.forward(
                    torch.cat((x, torch.tensor([lf_el])))[None, :]
                ) for x in coord_list_tensor])
                plt.plot(coord_list, mfacq_eval_high.detach().numpy(), label='high')
                plt.plot(coord_list, mfacq_eval_low.detach().numpy(), label='low')
                plt.legend()

            plt.title(problem_el.objective_function.name + '_acq_' + str(iteration))
            plt.tight_layout()

            plt.savefig(work_folder_name + '/' + exp_name + '/img/' + problem_el.objective_function.name + '/acq' + str(iteration) + '.png')

        iteration += 1

    x_names = ['x_' + str(i) for i in range(dim - 1)] + ['fid']

    if model_type_el != 'sogpr':
        x_low_rec = problem_el.optimize_mfacq_and_get_observation(mfacq, final=0)[0]
        x_low_rec[0][-1] = 1.0

        x_high_rec, y_high_rec, _ \
            = problem_el.optimize_mfacq_and_get_observation(mfacq, final=1)

        # print(problem_el.objective_function(x_low_rec), y_high_rec)
        # print(x_low_rec, x_high_rec)

        y_rec = np.minimum(
            problem_el.objective_function(x_low_rec),
            y_high_rec
        )

        if y_rec.flatten() == y_high_rec.flatten():
            x_rec = x_high_rec
        else:
            x_rec = x_low_rec

        df_rec = pd.DataFrame(
            np.hstack((x_rec, y_rec - y_min)),
            columns=x_names + ['y - y_min']
        )

    else:
        x_rec = train_x_high[torch.argmin(train_obj_high)]
        y_rec = torch.amin(train_obj_high)

        df_rec = pd.DataFrame(
            np.hstack((x_rec, y_rec - y_min))[None, :],
            columns=x_names[:-1] + ['y - y_min']
        )

    df = pd.DataFrame(columns=['cost', 'y - y_min'] + x_names,
                      data=np.hstack((np.array(opt_cost_history)[:, None],
                                      train_obj.numpy() - y_min,
                                      train_x.numpy(),)))
    df2 = pd.DataFrame(
        index=list(
            range(n_reg_init_el + (model_type_el != 'sogpr') * n_reg_lf_init_el - 1, len(df))),
        columns=['L1E', 'L2E', 'L1E/L1D', 'L1E/L2D', 'L2E/L1D', 'L2E/L2D'],
        data=np.array(rqmc_history)
    )

    df3 = pd.DataFrame(
        index=list(
            range(n_reg_init_el + (model_type_el != 'sogpr') * n_reg_lf_init_el - 1, len(df))),
        columns=['cir'],
        data=np.array(cir_history)
    )

    df_res = pd.concat([df, df2, df3], axis=1)

    print(DoE_no_path)

    df_rec.to_csv(DoE_no_path + '_rec.csv')
    df_res.to_csv(DoE_no_path + '.csv')

    return
