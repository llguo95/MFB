import pickle

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

def rqmc(pred_diff, samp_diff): # regression quality metric calculator
    L1E = np.abs(pred_diff).sum()  # L1 distance / prediction error
    L2E = np.sqrt((pred_diff ** 2).sum())  # L2 distance / prediction error

    L1D = np.abs(samp_diff).sum()  # L1 sample deviation (AAD / MAD)
    L2D = np.sqrt((samp_diff ** 2).sum())  # L2 sample deviation (Variance)

    L1E_div_L1D = L1E / L1D # "unexplained deviation" UD
    L2E_div_L2D = L2E / L2D # unexplained variance, (1 - R2) ** .5, FVU, UV

    L1E_div_L2D = L1E / (np.sqrt(len(pred_diff)) * L2D) # RMAE (RAAE)
    L2E_div_L1D = (np.sqrt(len(pred_diff)) * L2E) / L1D # RRMSE

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
                            train_x, train_obj, problem_el, model_type_el, dim, bds, noise_fix, lf_jitter, n_reg_el, optimize,
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
                                    model.models[cokg_dms_fid].update_model(False)  # do not call the underlying expensive algebra on load
                                    model.models[cokg_dms_fid].initialize_parameter()  # Initialize the parameters (connect the parameters up)
                                    model.models[cokg_dms_fid][:] = np.load(exp_path + ',' + str(cokg_dms_fid) + '.npy', allow_pickle=True)  # Load the parameters
                                    model.models[cokg_dms_fid].update_model(True)  # Call the algebra only once

                            #################################
                            ### Post-training; prediction ###
                            #################################
    
                            test_y_list_high, test_y_var_list_high, exact_y = posttrainer(
                                model, model_type_el, problem_el, bds, dim, n_inf, scaler_y_high
                            )

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
                                    plt.figure(num=problem_el.objective_function.name + 'DoE_no' + model_type_el, figsize=(4, 4))
                                    coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                    plt.plot(coord_list, test_y_list_high, 'r--', label='Predictive HF mean', alpha=.2,)
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

            vis_opt = 0
            if vis_opt:
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
