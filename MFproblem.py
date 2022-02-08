#################
#### Imports ####
#################
import numpy as np
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.kernels.linear_kernel import LinearKernel

from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy, qMultiFidelityMaxValueEntropy
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim.optimize import optimize_acqf
from botorch.optim.optimize import optimize_acqf_mixed

from botorch.models.multitask import MultiTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from cokgj import CoKrigingGP

#########################################
#### Problem & associated parameters ####
#########################################

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}
SMOKE_TEST = True  # os.environ.get("SMOKE_TEST")

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4


class MFProblem:
    def __init__(
            self,
            objective_function,
            cost_ratio=2,
            fidelities: Tensor = None,
    ):
        self.objective_function = objective_function
        self.fidelities = fidelities if fidelities is not None else torch.tensor([0.5, 1.0], **tkwargs)

        bounds = torch.tensor(objective_function._bounds, **tkwargs).transpose(0, 1)
        lf = self.fidelities[0]
        a = (1 - 1 / cost_ratio) / (1 - lf)
        cost_model = AffineFidelityCostModel(fidelity_weights={
            objective_function.dim - 1: a
        }, fixed_cost=1 - a)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        self.bounds = bounds
        self.cost_model = cost_model
        self.cost_aware_utility = cost_aware_utility

    ##########################
    #### Data initializer ####
    ##########################

    def generate_initial_data(self, n=16, n_lf=32, random=True, lf_mult=None, scramble=False, model_type='sogpr'):
        nl_exp = False
        nested_doe = True

        bds = self.objective_function._bounds

        if lf_mult is None:
            if model_type is not 'sogpr':
                n_tot = n + n_lf
            else:
                n_tot = n
        else:
            n_tot = n * lf_mult

        if random:
            train_x = torch.rand(n, self.objective_function.dim - 1, **tkwargs)

            p = 0.2 * torch.ones((n, 1))
            train_f = self.fidelities[torch.bernoulli(p).long()]
        else:
            soboleng = SobolEngine(dimension=self.objective_function.dim - 1, scramble=scramble)
            train_x = soboleng.draw(n_tot).to(**tkwargs)

            indices = torch.zeros((n_tot, 1))
            for i in range(len(indices)):
                if i < n:
                    indices[i] = 1
            train_f = self.fidelities[indices.long()]

        for x in train_x:
            for i in range(len(x)):
                x[i] = (bds[i][1] - bds[i][0]) * x[i] + bds[i][0]  # input scaling

        if nested_doe:
            train_x = torch.cat((train_x[:n], train_x[:n_tot - n]))

        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = self.objective_function(train_x_full).unsqueeze(-1)  # add output dimension

        if nl_exp:
            np.random.seed(42)
            X1 = np.linspace(0, 1, 50)[:, None]
            perm = np.random.permutation(50)
            X2 = X1[perm[0:14]]
            fid1 = np.zeros(X1.size)[:, None]
            fid2 = np.ones(X2.size)[:, None]
            train_x_full = torch.tensor(np.hstack((np.vstack((X1, X2)), np.vstack((fid1, fid2)))))
            train_obj = self.objective_function(train_x_full).unsqueeze(-1)

        return train_x_full, train_obj

    ##################################
    #### Regression model builder ####
    ##################################

    def initialize_model(self, train_x, train_obj, model_type='cokg', noise_fix=True):
        bds = None

        if model_type == 'cokg':
            model = CoKrigingGP(
                train_x,
                train_obj,
                input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
                outcome_transform=Standardize(m=1),
                noise_fix=noise_fix,
            )

        elif model_type == 'mtask':
            train_x[:, -1] = torch.floor(train_x[:, -1])
            model = MultiTaskGP(
                train_x,
                train_obj,
                task_feature=-1,
                input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
                outcome_transform=Standardize(m=1),
            )

        elif model_type == 'sogpr':
            n = torch.sum(train_x[:, -1] == 1)
            model = SingleTaskGP(
                train_x[:n, :-1],
                train_obj[:n],
                input_transform=Normalize(d=self.objective_function.dim - 1, bounds=bds),
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(LinearKernel())
            )

        else:
            model = SingleTaskMultiFidelityGP(
                train_x,
                train_obj,
                input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
                outcome_transform=Standardize(m=1),
                data_fidelity=self.objective_function.dim - 1,
                linear_truncated=False
            )

        if model_type == 'cokg_dms':
            mll = [ExactMarginalLogLikelihood(m.likelihood, m) for m in model]
        else:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

        ### Purgatory ###
        # elif model_type == 'nlcokg':
        #     model = NLCoKrigingGP(
        #         train_X=train_x,
        #         train_Y=train_obj,
        #         input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
        #         outcome_transform=Standardize(m=1),
        #     )

        # elif model_type == 'cokg_dms':
        #     models = CoKrigingGP_DMS(
        #         train_x,
        #         train_obj,
        #         input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
        #         outcome_transform=Standardize(m=1),
        #     )
        #     # model = [SingleTaskGP(
        #     #     train_x[self.lf_data_volume:],
        #     #     train_obj[self.lf_data_volume:],
        #     #     input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
        #     #     outcome_transform=Standardize(m=1),
        #     # )]
        #     # for i in range(1, 2):
        #     #     model.append(
        #     #         CoKrigingGP_DMS(
        #     #             train_x[:self.lf_data_volume],
        #     #             train_obj[:self.lf_data_volume],
        #     #             input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
        #     #             outcome_transform=Standardize(m=1),
        #     #             prev_model=model[i - 1],
        #     #         )
        #     #     )

        # elif model_type == 'mtask_icm':
        #     model = KroneckerMultiTaskGP(
        #         train_x,
        #         train_obj,
        #         input_transform=Normalize(d=self.objective_function.dim, bounds=bds),
        #         outcome_transform=Standardize(m=1),
        #     )

        return mll, model

    ##############################
    #### Acquisition function ####
    ##############################

    def get_mfacq(self, model):
        bounds_x = self.bounds
        candidate_set = torch.rand(16, bounds_x.size(1), device=self.bounds.device, dtype=self.bounds.dtype)

        train_f = self.fidelities[torch.randint(2, (16, 1))]
        candidate_set_full = torch.cat((candidate_set, train_f), dim=1)

        if model._get_name() is not 'SingleTaskGP':
            acq = qMultiFidelityMaxValueEntropy(
                model=model,
                candidate_set=candidate_set_full,
                num_fantasies=128 if not SMOKE_TEST else 2,
                cost_aware_utility=self.cost_aware_utility,
            )
        else:
            acq = qMaxValueEntropy(
                model=model,
                candidate_set=candidate_set,
                num_fantasies=128 if not SMOKE_TEST else 2,
            )

        return acq

    ###############################
    #### Acquisition optimizer ####
    ###############################

    def optimize_mfacq_and_get_observation(self, acq_f):

        cost = None
        if acq_f.model._get_name() is not 'SingleTaskGP':
            """Optimizes MFKG and returns a new candidate, observation, and cost."""
            aug_bounds = torch.hstack((self.bounds, torch.tensor([[0], [1]])))
            candidates, _ = optimize_acqf_mixed(
                acq_function=acq_f,
                bounds=aug_bounds,
                fixed_features_list=[{self.objective_function.dim - 1: float(self.fidelities[0])},
                                     {self.objective_function.dim - 1: 1.0}],
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={"batch_limit": 1, "maxiter": 5},
                sequential=True
            )
            # observe new values
            cost = self.cost_model(candidates).sum()
            # print(self.fidelities)
            # print(cost)
            new_x = candidates
            new_obj = self.objective_function(new_x).unsqueeze(-1)
        else:
            candidates, _ = optimize_acqf(
                acq_function=acq_f,
                bounds=self.bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES
            )
            new_x = candidates[0]
            new_obj = self.objective_function(torch.cat([new_x, torch.tensor([1])])).unsqueeze(-1)[:, None]
        return new_x, new_obj, cost

    #####################
    #### Recommender ####
    #####################

    def get_recommendation(self, model):
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=self.objective_function.dim,
            columns=[self.objective_function.dim - 1],
            values=[1],
        )

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=2,
            raw_samples=4,
            options={"batch_limit": 1, "maxiter": 5},
        )

        final_rec = rec_acqf._construct_X_full(final_rec)

        objective_value = self.objective_function(final_rec)
        return final_rec
