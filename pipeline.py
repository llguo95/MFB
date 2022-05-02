import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from botorch import fit_gpytorch_model
import GPy
import gpytorch
from emukit.multi_fidelity.models import NonLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels

from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

def scale_to_unit(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (el[d] - bds[0][d]) / (bds[1][d] - bds[0][d])
        el_c += 1
    return res

def pretrainer(
        problem_el,
        model_type_el,
        n_reg_el,
        n_reg_lf_el,
        lf_el,
        random,
        scramble,
        n_inf,
        bds,
        dim,
):
    problem_el.fidelities = torch.tensor([lf_el, 1.0], **tkwargs)

    train_x, train_obj = problem_el.generate_initial_data(
        n=n_reg_el,
        n_lf=n_reg_lf_el,
        random=random,
        scramble=scramble,
        model_type=model_type_el,
    )

    train_x_low, train_x_high = [], []
    train_y_low, train_y_high = [], []
    for i in range(len(train_x)):
        if train_x[i][-1] != 1.0:
            train_x_low.append(train_x[i][:-1].cpu().numpy())
            train_y_low.append(train_obj[i].cpu().numpy())
        else:
            train_x_high.append(train_x[i][:-1].cpu().numpy())
            train_y_high.append(train_obj[i].cpu().numpy())
    train_x_low, train_x_high = np.array(train_x_low), np.array(train_x_high)
    train_y_low, train_y_high = np.array(train_y_low), np.array(train_y_high)

    bds = bds.cpu()

    test_x_axes = [np.linspace(bds[0, k], bds[1, k], int(n_inf ** (1 / (dim - 1)))) for k in range(dim - 1)]

    test_x = np.meshgrid(*test_x_axes)

    test_x_list = np.hstack([layer.reshape(-1, 1) for layer in test_x])

    lf_vec = problem_el.fidelities[0].cpu() * np.ones((np.shape(test_x_list)[0], 1))
    hf_vec = np.ones((np.shape(test_x_list)[0], 1))

    test_x_list_low = np.concatenate((test_x_list, lf_vec), axis=1)
    test_x_list_high = np.concatenate((test_x_list, hf_vec), axis=1)

    exact_y = problem_el.objective_function(torch.Tensor(test_x_list_high)).cpu().detach().numpy()
    exact_y_low = problem_el.objective_function(torch.Tensor(test_x_list_low)).cpu().detach().numpy()

    ### Scaling ###
    test_x_list_scaled = None
    scaler_y_high = None
    scaler_y_low = None
    if model_type_el in ['cokg_dms', 'nlcokg']:
        train_x_low_scaled = scale_to_unit(train_x_low, bds)
        train_x_high_scaled = scale_to_unit(train_x_high, bds)

        scaler_y_low = StandardScaler()
        scaler_y_high = StandardScaler()

        exact_y_low = problem_el.objective_function(torch.Tensor(test_x_list_low)).cpu().detach().numpy()
        scaler_y_low.fit(exact_y_low[:, None])
        scaler_y_high.fit(exact_y[:, None])
        train_y_low_scaled = scaler_y_low.transform(train_y_low)
        train_y_high_scaled = scaler_y_high.transform(train_y_high)

        test_x_list_scaled = scale_to_unit(test_x_list, bds)

        train_x = [train_x_low_scaled, train_x_high_scaled]
        train_obj = [train_y_low_scaled, train_y_high_scaled]

    return train_x, train_y_high, train_obj, test_x_list, test_x_list_scaled, test_x_list_high, scaler_y_high, exact_y, exact_y_low, train_y_low, scaler_y_low,

def trainer(
        train_x,
        train_obj,
        problem_el,
        model_type_el,
        dim,
        noise_fix,
        lf_jitter,
):
    if model_type_el == 'cokg_dms':
        base_k = GPy.kern.RBF
        kernels_RL = [base_k(dim - 1) + GPy.kern.White(dim - 1), base_k(dim - 1)]
        model = GPy.models.multiGPRegression(
            train_x,
            train_obj,
            kernel=kernels_RL,
        )

        if noise_fix:
            model.models[1].Gaussian_noise.variance.fix(lf_jitter)
        model.optimize()

    elif model_type_el == 'nlcokg':

        train_x_low_scaled_aug = np.hstack((train_x[0], np.zeros(len(train_x[0]))[:, None]))
        train_x_high_scaled_aug = np.hstack((train_x[1], np.ones(len(train_x[1]))[:, None]))

        train_X_scaled = np.vstack((train_x_low_scaled_aug, train_x_high_scaled_aug))

        train_Y_scaled = np.vstack((train_x[0], train_x[1]))

        base_kernel = GPy.kern.RBF
        kernels = make_non_linear_kernels(base_kernel, 2, dim - 1)

        model = NonLinearMultiFidelityModel(train_X_scaled,
                                            train_Y_scaled,
                                            n_fidelities=2, kernels=kernels,
                                            verbose=False, optimization_restarts=10
                                            )
        if noise_fix:
            model.models[1].Gaussian_noise.variance.fix(lf_jitter)
        model.optimize()

    else:
        mll, model = problem_el.initialize_model(train_x, train_obj,
                                                 model_type=model_type_el, noise_fix=noise_fix)

        # print('PRE LIKELIHOOD', model.likelihood)
        if noise_fix:
            model.likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 2e-4))
            # cons = gpytorch.constraints.constraints.Interval(1e-4, 2e-4)
            # model.likelihood.noise_covar.register_constraint("raw_noise", cons)
        else:
            model.likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        print('POST LIKELIHOOD', model.likelihood)

        fit_gpytorch_model(mll)
    return model

from scipy.stats import norm

def ei(mean_x, var_x, f_inc):
    mean_x = -mean_x  # minimization
    Delta = mean_x - f_inc
    std = np.sqrt(np.abs(var_x))
    res = np.maximum(Delta, np.zeros(Delta.shape)) \
          + std * norm.pdf(Delta / std) \
          - np.abs(Delta) * norm.cdf(Delta / std)
    return res


def posttrainer(
        model,
        model_type_el,
        test_x_list,
        test_x_list_scaled,
        test_x_list_high,
        scaler_y_high,
        scaler_y_low,
):
    test_y_list_high_scaled = None
    test_y_list_low = None
    test_y_var_list_low = None
    if model_type_el == 'cokg_dms':
        pred_mu, pred_sigma = model.predict(test_x_list_scaled)
        test_y_list_high = scaler_y_high.inverse_transform(pred_mu[1])
        test_y_var_list_high = scaler_y_high.inverse_transform(pred_sigma[1])
        # test_y_list_high = scaler_y_high.inverse_transform(pred_mu)
        # test_y_var_list_high = scaler_y_high.inverse_transform(pred_sigma)

        test_y_list_low = scaler_y_low.inverse_transform(pred_mu[0])
        test_y_var_list_low = scaler_y_low.inverse_transform(pred_sigma[0])

        # test_y_list_high_scaled = pred_mu[1]

    elif model_type_el == 'nlcokg':
        test_x_aug = np.hstack((test_x_list_scaled, np.ones(test_x_list.shape)))
        pred_mu, pred_sigma = model.predict(test_x_aug)
        test_y_list_high = scaler_y_high.inverse_transform(pred_mu)
        test_y_var_list_high = scaler_y_high.inverse_transform(pred_sigma)

    elif model_type_el == 'mtask':
        test_y_list_high = model.posterior(torch.from_numpy(test_x_list)).mean.detach().numpy()[:, 1][:, None]
        test_y_var_list_high = model.posterior(torch.from_numpy(test_x_list)).variance.detach().numpy()[:, 1][:, None]

        test_y_list_low = model.posterior(torch.from_numpy(test_x_list)).mean.detach().numpy()[:, 0][:, None]
        test_y_var_list_low = model.posterior(torch.from_numpy(test_x_list)).variance.detach().numpy()[:, 0][:, None]
        #
        # test_y_list_high_scaled = \
        #     model.outcome_transform(model.posterior(torch.from_numpy(test_x_list).to(**tkwargs)).mean)[0].detach().numpy()[:, 1]

    elif model_type_el == 'sogpr':
        test_y_list_high = model.posterior(torch.from_numpy(test_x_list).to(**tkwargs)).mean.cpu().detach().numpy()
        test_y_var_list_high = model.posterior(torch.from_numpy(test_x_list).to(**tkwargs)).mvn.covariance_matrix.diag().cpu().detach().numpy()[:, None]

        # test_y_list_high_scaled = model.outcome_transform(model.posterior(torch.from_numpy(test_x_list).to(**tkwargs)).mean)[0].detach().numpy()

    else:
        test_y_list_high = model.posterior(torch.from_numpy(test_x_list_high).to(**tkwargs)).mean.cpu().detach().numpy()
        test_y_var_list_high = model.posterior(
            torch.from_numpy(test_x_list_high).to(**tkwargs)).mvn.covariance_matrix.diag().cpu().detach().numpy()[:, None]

        # test_y_list_high_scaled = \
        #     model.outcome_transform(model.posterior(torch.from_numpy(test_x_list_high).to(**tkwargs)).mean)[0].detach().numpy()
        #
        test_y_list_low = model.posterior(torch.from_numpy(test_x_list)).mean.detach().numpy()[:, 0][:, None]
        test_y_var_list_low = model.posterior(torch.from_numpy(test_x_list)).mvn.covariance_matrix.diag().cpu().detach().numpy()[:, None]

    return test_y_list_high, test_y_var_list_high, test_y_list_high_scaled, test_y_list_low, test_y_var_list_low


def acq_visualizer(_, test_x_list, test_y_list_high, test_y_var_list_high, exact_y, acq_ei_norm, acq_ucb_norm):
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

def reg_main_visualizer(_, test_x_list, test_y_list_high, test_y_var_list_high, exact_y, acq_ei_norm, acq_ucb_norm):
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
