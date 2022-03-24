import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from gpytorch.constraints.constraints import Interval

from botorch import fit_gpytorch_model
from pybenchfunction import function

from scipy.stats.mstats import gmean

from objective_formatter import botorch_TestFunction, AugmentedTestFunction

from MFproblem import MFProblem
from pipeline import pretrainer, trainer, posttrainer

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

def reg_main(
        problem=None, model_type=None, lf=None, n_reg=None, n_reg_lf=None, scramble=False, random=False,
        n_inf=500, noise_fix=True, lf_jitter=1e-4, noise_type='b',
):
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

    metadata_dict = {
        'problem': [p.objective_function.name for p in problem],
        'model_type': model_type,
        'lf': lf,
        'n_reg': n_reg,
        'n_reg_lf': n_reg_lf,
        'scramble': scramble,
        'noise_fix': noise_fix,
        'noise_type': noise_type,
    }

    model_type_data = {}
    for model_type_el in model_type:
        print()
        print(model_type_el)

        problem_data = {}
        for problem_el in problem:
            print()
            print(problem_el.objective_function.name)

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim

            lf_data = {}
            for lf_el in lf:
                print()
                print('lf =', lf_el)

                n_reg_data = {}
                RAAE_stats_dict = {}
                RMSTD_stats_dict = {}
                ei_stats_dict = {}
                ucb_stats_dict = {}
                for n_reg_el, n_reg_lf_el in zip(n_reg, n_reg_lf):
                    print('n_reg =', n_reg_el)
                    print('n_reg_lf_el =', n_reg_lf_el)

                    n_DoE_RAAE_data = []
                    n_DoE_RMSTD_data = []
                    n_DoE_x_rec_data = {
                        'ei': [],
                        'ucb': [],
                    }
                    n_DoE_y_rec_data = {
                        'ei': [],
                        'ucb': [],
                    }

                    for _ in range(n_DoE):
                        ####################
                        ### Pre-training ###
                        ####################

                        (train_x, train_y_high, train_obj, test_x_list, test_x_list_scaled, test_x_list_high,
                         scaler_y_high, exact_y, ) = pretrainer(
                            problem_el, model_type_el, n_reg_el, n_reg_lf_el, lf_el, random, scramble, n_inf, bds, dim,
                        )

                        ################
                        ### Training ###
                        ################

                        model = trainer(train_x, train_obj, problem_el, model_type_el, dim, noise_fix, lf_jitter, )

                        #################################
                        ### Post-training; prediction ###
                        #################################

                        (test_y_list_high, test_y_var_list_high, ) = posttrainer(
                            model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high,
                        )

                        ########################
                        ### Post-processing ####
                        ########################

                        vis2d = True
                        if vis2d:
                            if dim - 1 == 2:
                                # print(bds)
                                coord_mesh, _ = uniform_grid(bl=bds[0], tr=bds[1], n=[22, 22], mesh=True)
                                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                                ax.plot_surface(coord_mesh[0], coord_mesh[1], test_y_list_high.reshape(coord_mesh[0].shape), cmap='viridis', linewidth=0, alpha=.5)
                                ax.scatter(train_x[:n_reg_el][:, 0], train_x[:n_reg_el][:, 1], train_y_high, c='r', s=50)
                                plt.tight_layout()

                                print(model.covar_module.base_kernel)
                            elif dim - 1 == 1:
                                plt.figure()
                                coord_list = uniform_grid(bl=bds[0], tr=bds[1], n=[500])
                                plt.plot(coord_list, test_y_list_high)
                                plt.scatter(train_x[:n_reg_el][:, 0], train_y_high, c='r')
                                plt.tight_layout()

                        RAAE = np.mean(np.abs(exact_y.reshape(np.shape(test_y_list_high)) - test_y_list_high)) / np.std(
                            exact_y)

                        RMSTD = np.mean(np.sqrt(np.abs(test_y_var_list_high))) / (max(exact_y) - min(exact_y))

                        acq_ei = ei(
                            mean_x=test_y_list_high,
                            var_x=test_y_var_list_high,
                            f_inc=np.amax(train_y_high)
                        )

                        acq_ucb = ucb(
                            mean_x=test_y_list_high,
                            var_x=test_y_var_list_high,
                            kappa=2
                        )

                        if max(acq_ei) != min(acq_ei):
                            acq_ei_norm = (acq_ei - min(acq_ei)) / (max(acq_ei) - min(acq_ei))
                        else:
                            acq_ei_norm = acq_ei

                        if max(acq_ucb) != min(acq_ucb):
                            acq_ucb_norm = (acq_ucb - min(acq_ucb)) / (max(acq_ucb) - min(acq_ucb))
                        else:
                            acq_ucb_norm = acq_ucb

                        if vis:
                            if _ == 0:
                                fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all')

                            axs[0].plot(test_x_list, test_y_list_high, 'k--', alpha=.25)
                            axs[0].fill_between(test_x_list.flatten(),
                                                (test_y_list_high - 2 * np.sqrt(
                                                    np.abs(test_y_var_list_high))).flatten(),
                                                (test_y_list_high + 2 * np.sqrt(
                                                    np.abs(test_y_var_list_high))).flatten(),
                                                color='k', alpha=.05)

                            axs[0].plot(test_x_list, exact_y, 'k', linewidth=.5)
                            axs[0].scatter(
                                test_x_list[np.argmin(exact_y)], np.amin(exact_y), marker='o', c='r',
                                zorder=np.iinfo(np.int32).max
                            )
                            axs[0].set_title('Objective')

                            c = (np.amax(exact_y) - np.amin(exact_y)) / 5
                            axs[0].set_ylim([np.amin(exact_y) - c, np.amax(exact_y) + c])

                            axs[1].scatter(
                                test_x_list, acq_ei_norm, marker='s', facecolors='none', edgecolors='k',
                                s=10, alpha=.2
                            )

                            axs[1].set_title('EI')

                            axs[1].scatter(
                                test_x_list[np.argmax(acq_ei_norm)], np.amax(acq_ei_norm), marker='s',
                                facecolors='none', edgecolors='r',
                                s=30, zorder=np.iinfo(np.int32).max
                            )

                            axs[2].scatter(
                                test_x_list, acq_ucb_norm, marker='d', facecolors='none', edgecolors='k',
                                s=10, alpha=.2
                            )

                            axs[2].scatter(
                                test_x_list[np.argmax(acq_ucb_norm)], np.amax(acq_ucb_norm), marker='d',
                                facecolors='none', edgecolors='r',
                                s=30, zorder=np.iinfo(np.int32).max
                            )

                            axs[2].set_title('UCB')

                            plt.tight_layout()

                        x_next_ei = test_x_list[np.argmax(acq_ei_norm)]
                        y_next_ei = problem_el.objective_function(torch.tensor([x_next_ei[0], 1])).cpu().detach().numpy()

                        x_next_ucb = test_x_list[np.argmax(acq_ucb_norm)]
                        y_next_ucb = problem_el.objective_function(torch.tensor([x_next_ucb[0], 1])).cpu().detach().numpy()

                        n_DoE_RAAE_data.append(RAAE)
                        n_DoE_RMSTD_data.append(RMSTD)

                        x_opt, y_opt = problem_el.objective_function.opt(d=dim - 1)

                        n_DoE_x_rec_data['ei'].append(x_next_ei)
                        n_DoE_x_rec_data['ucb'].append(x_next_ucb)

                        n_DoE_y_rec_data['ei'].append(np.abs(y_next_ei - y_opt) / (np.amax(exact_y) - y_opt))
                        n_DoE_y_rec_data['ucb'].append(np.abs(y_next_ucb - y_opt) / (np.amax(exact_y) - y_opt))

                    RAAE_stats_dict['amean'] = np.mean(n_DoE_RAAE_data)
                    RAAE_stats_dict['gmean'] = gmean(n_DoE_RAAE_data)
                    RAAE_stats_dict['median'] = np.median(n_DoE_RAAE_data)
                    RAAE_stats_dict['quantiles'] = [np.quantile(n_DoE_RAAE_data, q=.25), np.quantile(n_DoE_RAAE_data, q=.75)]

                    RMSTD_stats_dict['amean'] = np.mean(n_DoE_RMSTD_data)
                    RMSTD_stats_dict['gmean'] = gmean(n_DoE_RMSTD_data)
                    RMSTD_stats_dict['median'] = np.median(n_DoE_RMSTD_data)
                    RMSTD_stats_dict['quantiles'] = [np.quantile(n_DoE_RMSTD_data, q=.25), np.quantile(n_DoE_RMSTD_data, q=.75)]

                    ei_stats_dict['median'] = np.median(n_DoE_y_rec_data['ei'])
                    ei_stats_dict['quantiles'] = [np.quantile(n_DoE_y_rec_data['ei'], q=.25), np.quantile(n_DoE_y_rec_data['ei'], q=.75)]

                    ucb_stats_dict['median'] = np.median(n_DoE_y_rec_data['ucb'])
                    ucb_stats_dict['quantiles'] = [np.quantile(n_DoE_y_rec_data['ucb'], q=.25), np.quantile(n_DoE_y_rec_data['ucb'], q=.75)]

                    n_reg_data[(n_reg_el, n_reg_lf_el)] = {
                        'RAAE_stats': RAAE_stats_dict.copy(),
                        'RMSTD_stats': RMSTD_stats_dict.copy(),
                        'ei_stats': ei_stats_dict.copy(),
                        'ucb_stats': ucb_stats_dict.copy(),
                    }

                lf_data[lf_el] = n_reg_data
                # print(n_reg_data)
            problem_data[problem_el.objective_function.name] = lf_data
        model_type_data[model_type_el] = problem_data

    return model_type_data, metadata_dict

def scale_to_unit(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (el[d] - bds[0][d]) / (bds[1][d] - bds[0][d])
        el_c += 1
    return res

def bo_main(problem=None, model_type=None, lf=None, n_reg_init=None, scramble=True,
            n_inf=500, random=False, noise_fix=True, n_reg_lf_init=None, budget=25,):
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
        'noise_type': [p.objective_function.noise_type for p in problem]
    }

    model_type_data = {}
    for model_type_el in model_type:

        problem_data = {}
        for problem_el in problem:
            if vis:
                plt.figure(num=problem_el.objective_function.name)
            pm_one = 2 * (.5 - problem_el.objective_function.negate)

            bds = problem_el.bounds
            dim = problem_el.objective_function.dim
            # xmin, ymin = problem_el.objective_function.opt(d=dim - 1)

            lf_data = {}
            for lf_el in lf:

                n_reg_init_data = {}
                for n_reg_init_el, n_reg_lf_init_el in zip(n_reg_init, n_reg_lf_init):
                    problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)

                    (train_x, train_y_high, train_obj,
                     test_x_list, test_x_list_scaled, test_x_list_high,
                     scaler_y_high, exact_y) = pretrainer(
                        problem_el, model_type_el, n_reg_init_el, n_reg_lf_init_el, lf_el, random,
                        scramble, n_inf, bds, dim,
                    )

                    train_x_init, train_y_init = train_x, train_obj

                    train_x_high = torch.tensor([])
                    train_obj_high = torch.tensor([])

                    cumulative_cost = 0.0 # change into TOTAL cost (+ intiial DoE cost)

                    opt_data = {}
                    _ = 0
                    RAAEs, RMSTDs = [], []
                    while cumulative_cost < budget:
                        if _ % 5 == 0: print(_)
                        mll, model = problem_el.initialize_model(train_x, train_obj, model_type=model_type_el)
                        if noise_fix:
                            cons = Interval(1e-4, 1e-4 + 1e-10)
                            model.likelihood.noise_covar.register_constraint("raw_noise", cons)
                        fit_gpytorch_model(mll)

                        (test_y_list_high, test_y_var_list_high) = posttrainer(
                            model, model_type_el, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high,
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
                            cumulative_cost += cost

                            if new_x[0][-1] == 1:
                                train_x_high = torch.cat([train_x_high, new_x])
                                train_obj_high = torch.cat([train_obj_high, new_obj])
                        else:
                            new_x = torch.cat([new_x, torch.tensor([1])])[:, None].T
                            train_x = torch.cat([train_x, new_x])
                            train_obj = torch.cat([train_obj, new_obj])
                            cumulative_cost += 1

                            train_x_high = torch.cat([train_x_high, new_x])
                            train_obj_high = torch.cat([train_obj_high, new_obj])

                        # if vis:
                        #     size = 2 * (_ + 1)
                        #     # plt.figure(model_type_el)
                        #     if model_type_el is not 'sogpr':
                        #         if new_x[0][-1] == 1:
                        #             print('Sample at highest fidelity!')
                        #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
                        #                         color='g')
                        #         else:
                        #             # continue
                        #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
                        #                         color='b')
                        #     else:
                        #         plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
                        #                     color='g')
                        #     if cumulative_cost + cost > budget:
                        #         plt.plot(test_x_list, pm_one * test_y_list_high, alpha=.1,
                        #                  color='r')
                        #         plt.fill_between(test_x_list.flatten(),
                        #                          (pm_one * test_y_list_high - 2 * np.sqrt(
                        #                              np.abs(test_y_var_list_high))).flatten(),
                        #                          (pm_one * test_y_list_high + 2 * np.sqrt(
                        #                              np.abs(test_y_var_list_high))).flatten(),
                        #                          color='k', alpha=.05 #* (_ + 1) / n_iter_el
                        #                          )
                        #         if new_x[0][-1] == 1:
                        #             print('Sample at highest fidelity!')
                        #             plt.scatter(new_x[0][0], pm_one * new_obj[0][0], s=size,
                        #                         color='orange')
                        #     plt.plot(test_x_list, pm_one * exact_y, 'k--')
                        _ += 1

                    # print(problem_el.get_recommendation(model))

                    # print(train_x_init)

                    # if len(train_obj_high) > 0:
                    #     # print('Initial DoE EXCLUDED in calculating cumulative minimum')
                    #     x_best = train_x_high[torch.argmax(train_obj_high)]
                    #     y_best = -train_obj_high[torch.argmax(train_obj_high)]
                    # else:
                    #     # print('Initial DoE INCLUDED in calculating cumulative minimum')
                    #     x_best = train_x[torch.argmax(train_obj)]
                    #     y_best = -train_obj[torch.argmax(train_obj)]
                    #
                    # opt_data_vol = _ - 1
                    #
                    # x_err_norm = np.linalg.norm(x_best[:-1] - xmin) / np.linalg.norm(problem_el.bounds[1] - problem_el.bounds[0])
                    # y_err_norm = np.linalg.norm(y_best - ymin) / (np.amax(-exact_y) - ymin)
                    #
                    # x_hist_norm = train_x[:, :-1]
                    # y_hist_norm = np.linalg.norm(-train_obj - ymin, axis=1) / (np.amax(-exact_y) - ymin)

                    opt_data['x_hist'] = train_x.detach().numpy()
                    # print(problem_el.objective_function.negate)
                    opt_data['y_hist'] = pm_one * train_obj
                    opt_data['RAAE'] = torch.tensor(RAAEs).to(**tkwargs)[:, None]
                    opt_data['RMSTD'] = torch.tensor(RMSTDs).to(**tkwargs)[:, None]
                    # opt_data['y_hist_norm'] = np.hstack((y_hist_norm[:, None], train_x[:, -1].detach().numpy()[:, None]))

                    n_reg_init_data[(n_reg_init_el, n_reg_lf_init_el)] = opt_data

                lf_data[lf_el] = n_reg_init_data

            problem_data[problem_el.objective_function.name] = lf_data

        model_type_data[model_type_el] = problem_data

    return model_type_data, metadata_dict