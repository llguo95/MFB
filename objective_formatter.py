import torch
from torch import Tensor
import numpy as np
from botorch.test_functions import SyntheticTestFunction

class botorch_TestFunction(SyntheticTestFunction):
    def __init__(self, fun, negate=False):
        self.name = fun.name
        self.continuous = fun.continuous
        self.convex = fun.convex
        self.separable = fun.separable
        self.differentiable = fun.differentiable
        self.multimodal = fun.multimodal
        self.randomized_term = fun.randomized_term
        self.parametric = fun.parametric

        self.fun = fun
        self.dim = fun.d
        self._bounds = fun.input_domain
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        self.negate = negate
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.Tensor(
            np.apply_along_axis(
                self.fun, 1, X
            )
        )

        if self.negate:
            res = -res

        return res

class AugmentedTestFunction(SyntheticTestFunction):
    def __init__(self, fun, abs=False, noise_type='bn'):
        self.name = 'Augmented' + fun.name.replace(' ', '')
        self.continuous = fun.continuous
        self.convex = fun.convex
        self.separable = fun.separable
        self.differentiable = fun.differentiable
        self.multimodal = fun.multimodal
        self.randomized_term = fun.randomized_term
        self.parametric = fun.parametric

        self.fun = fun.evaluate_true
        self.opt = fun.fun.get_global_minimum
        self.dim = fun.dim + 1
        self._bounds = fun._bounds
        self._optimizers = fun._optimizers

        self.noise_type = noise_type

        # self.abs = abs
        super().__init__()

    def evaluate_true(self, X: Tensor) -> Tensor:
        # torch.random.manual_seed(123)
        # white_noise_pre = torch.normal(0, 1, size=(2000,))

        res_high = self.fun(X[:, :-1]).flatten()

        fid = X[:, -1]
        noise = 2 * (torch.rand(res_high.shape) - 0.5)
        white_noise = torch.normal(0, 1, size=(len(res_high),))
        # white_noise = white_noise_pre[:len(res_high)]
        brown_noise = torch.cumsum(white_noise, dim=-1)

        stdev = 1
        if len(res_high) > 1:
            stdev = torch.std(res_high)

        if self.noise_type == 'bn':
            res_low = stdev * white_noise + torch.mean(res_high) #+ 500 * brown_noise
        elif self.noise_type == 'n':
            res_low = stdev * white_noise + res_high
        elif self.noise_type == 'b':
            res_low = torch.mean(res_high)
        else:
            res_low = stdev * white_noise + torch.mean(res_high) #+ 500 * brown_noise

        # res_low = torch.mean(res_high) #+ 500 * brown_noise

        # res_low = self.fun(X[:, :-1] - 5).flatten()

        # res_low = res_high * 1.25 #+ torch.mean(res_high) #+ 500 * brown_noise
        # res_low = np.sqrt(bds[0][1] - bds[0][0]) * brown_noise + torch.mean(res_high)
        c = 1

        res = fid ** c * res_high + (1 - fid ** c) * res_low

        # if self.abs and fid[0] != 1:
        #     res = torch.abs(res)

        # if self.negate:
        #     res = -res

        return res