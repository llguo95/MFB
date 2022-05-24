import numpy as np

import pybenchfunction
import matplotlib.pyplot as plt

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
# fig, axs = plt.subplots(3, 3)
# for f_no, f in enumerate(f_class_list[:9]):
#     a_id = (f_no % 9) // 3, f_no % 3
#     pybenchfunction.plot_2d(f(d=2), n_space=100, show=False, ax=axs[a_id])
#     axs[a_id].set_title(f.name)

f_list = [f.name for f in f_class_list]

excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
f_class_list = [f for f in f_class_list if f.name not in excluded_fs]
print(list(enumerate([f.name for f in f_class_list])))

dim = 1
for f_no, fun in enumerate(f_class_list):
    f = fun(d=dim)
    fmin = f.get_global_minimum(d=dim)
    if f_no % 9 == 0:
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    a_id = (f_no % 9) // 3, f_no % 3
    axs[a_id].set_title(f.name)
    if dim == 1:
        x = np.linspace(f.input_domain[0][0], f.input_domain[0][1], 500)[:, None]
        y = np.apply_along_axis(f, 1, x)
        axs[a_id].plot(x, y)
        # print(f.name)
        # print(fmin)
        axs[a_id].scatter(fmin[0], fmin[1], c='red', marker='*', s=100, zorder=10)
        axs[a_id].grid()
    if dim == 2:
        pybenchfunction.plot_2d(f, n_space=100, show=False, ax=axs[a_id])
        # print(f.name)
        # print(fmin)
        if f.name == 'Qing':
            axs[a_id].scatter(fmin[0][:, 0], fmin[0][:, 1], c='red', marker='*', s=100, zorder=10)
        else:
            axs[a_id].scatter(fmin[0][0], fmin[0][1], c='red', marker='*', s=100, zorder=10)
    plt.tight_layout()
plt.show()