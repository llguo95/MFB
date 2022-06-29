import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
import pybenchfunction

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)

f_list = [f.name for f in f_class_list]

excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
f_class_list = [f for f in f_class_list if f.name not in excluded_fs]

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

def scale_to_orig(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (bds[d][1] - bds[d][0]) * el[d] + bds[d][0]
        el_c += 1
    return res

max_data = {}
for d in [1, 2, 3, 4]:
    sobolsamp = Sobol(d=d)
    maxs = []
    for f in f_class_list:
        bds = f(d=d).input_domain
        x = scale_to_orig(sobolsamp.random(round(50 / d) ** d), bds)

        y = np.apply_along_axis(
            func1d=f(d=d),
            axis=1,
            arr=x
        )
        maxs.append(np.amax(y))
    max_data[d] = maxs
df = pd.DataFrame(max_data, index=[f.name.replace(' ', '') for f in f_class_list])
print(df)
df.to_csv('max_vals.csv')