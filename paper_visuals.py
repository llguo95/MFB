import numpy as np
import torch
from matplotlib import pyplot as plt

import pybenchfunction
from MFproblem import MFProblem
from objective_formatter import AugmentedTestFunction, botorch_TestFunction

from torch.quasirandom import SobolEngine

def scale_to_unit(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (el[d] - bds[0][d]) / (bds[1][d] - bds[0][d])
        el_c += 1
    return res

def scale_to_orig(x, bds):
    res = torch.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = el[d] * (bds[1][d] - bds[0][d]) + bds[0][d]
        el_c += 1
    return res

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]

fig_num = [1]

if 1 in fig_num:
    ls_dict = {'b': '--', 'n': ''}
    marker_dict = {'b': ' ', 'n': '.'}
    c_dict = {.1: 'moccasin', .5: 'orange', .9: 'orangered'}
    for noise_type in ['b', 'n']:
        AlpineN2_1d = MFProblem(
            objective_function=AugmentedTestFunction(
                botorch_TestFunction(
                    fs[0](d=1),
                ), negate=False, noise_type=noise_type
            ).to(**tkwargs)
        )

        bds = AlpineN2_1d.bounds

        soboleng = SobolEngine(dimension=1)
        x_unit = soboleng.draw(256).to(**tkwargs)
        x, _ = torch.sort(scale_to_orig(x_unit, bds), dim=0)
        x_hf = torch.hstack((x, torch.ones_like(x)))
        y_hf = AlpineN2_1d.objective_function(x_hf)

        plt.figure(num=noise_type + '-Augmented function showcase')
        plt.plot(x, y_hf, c='brown', label='High fidelity')

        AlpineN2_1d.noise_type = 'n'

        for lf in [0.1, 0.5, 0.9]:
            x_lf = torch.hstack((x, lf * torch.ones_like(x)))
            y_lf = AlpineN2_1d.objective_function(x_lf)
            plt.plot(x, y_lf,
                     # alpha=.5 * lf + .25,
                     color=c_dict[lf],
                     linestyle=ls_dict[noise_type], marker=marker_dict[noise_type], label='LF = ' + str(lf))

        plt.ylim([-5, 30])

        plt.legend()
        plt.tight_layout()

plt.show()