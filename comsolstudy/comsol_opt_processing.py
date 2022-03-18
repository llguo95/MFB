import pickle

import matplotlib.pyplot as plt

from MFBO import Comsol_Sim_high, Comsol_Sim_low

import numpy as np

from sklearn.metrics import r2_score

folder_path = 'data/'

file_names = [
    # 1D
    # '20220301151816', # stmf
    # '20220301152309', # sogpr
    # '20220301163625', # stmf
    # '20220301164020', # sogpr

    # 2D
    # '20220301165310',  # stmf # IDENTICAL FUNCTIONS
    # '20220301164441',  # stmf
    # '20220301164506',  # sogpr
    # '20220301164642',  # cokg

    # 2D Styblinsky-Tang
    # '20220303171503', # sogpr
    # '20220303171823', # stmf
    # '20220303171947', # cokg

    # # 2D Styblinsky-Tang
    # '20220303172116', # sogpr
    # '20220303172321', # stmf
    # '20220303172447', # cokg

    # NEW 2D Styblinsky-Tang
    # '20220316202202',
    '20220316221812'
]

colors = {
    'sogpr': 'r',
    'cokg': 'b',
    'stmf': 'g',
}

### calculating function similarity ###
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

x = uniform_grid([0, 0], [1, 1], [100, 100])
y_h = np.array([Comsol_Sim_high(x_i) for x_i in x])
y_l = np.array([Comsol_Sim_low(x_i) for x_i in x])
# print(y_h, y_l)
RAAE = np.mean(np.abs(y_h - y_l)) / np.std(y_h)
print('RAAE:', RAAE)
print('R^2 score:', r2_score(y_h, y_l))

### plotting data ###

for f_i, file_name in enumerate(file_names):
    # print(file_name)
    open_file = open(folder_path + file_name + '.pkl', 'rb')
    data = pickle.load(open_file)
    open_file.close()

    # print(data)

    open_file = open(folder_path + file_name + '_metadata.pkl', 'rb')
    metadata = pickle.load(open_file)
    open_file.close()

    # print(metadata['budget'])

    # model_hists = {}
    for model_type in metadata['model_type']:

        y_hist_norm_agg = []
        for i in range(len(data)):
            model_type_slice = data[i][model_type]

            for problem in metadata['problem']:
                problem_slice = model_type_slice[problem]

                for lf in metadata['lf']:
                    lf_slice = problem_slice[lf]

                    for n_reg_init, n_reg_lf_init in zip(metadata['n_reg_init'], metadata['n_reg_lf_init']):
                        n_reg_slice = lf_slice[(n_reg_init, n_reg_lf_init)]

                        if model_type == 'sogpr':
                            n_reg_lf_init = 0

                        x_hist = n_reg_slice['x_hist']
                        fid_hist = x_hist[:, -1]
                        y_hist = n_reg_slice['y_hist']
                        # print(fid_hist, y_hist)
                        y_hist_init, y_hist_opt = y_hist[:n_reg_init + (model_type != 'sogpr') * n_reg_lf_init], y_hist[n_reg_init + (model_type != 'sogpr') * n_reg_lf_init:]
                        fid_hist_init, fid_hist_opt = fid_hist[:n_reg_init + (model_type != 'sogpr') * n_reg_lf_init], fid_hist[n_reg_init + (model_type != 'sogpr') * n_reg_lf_init:]
                        # print(y_hist_init)

                        y_hist_cum = []
                        cost_cum = [0]
                        cost_cum_high = [0]
                        y_min_inc = 1e20 # large initialization value

                        for j, x_j in enumerate(x_hist):
                            if x_j[-1] == 1:
                                y_min_inc = min(y_min_inc, Comsol_Sim_high(x_j[:-1]))
                                y_hist_cum.append(y_min_inc)
                                cost_cum.append(cost_cum[-1] + 1)
                                cost_cum_high.append(cost_cum[-1])
                            else:
                                cost_cum.append(cost_cum[-1] + 1 / metadata['cost_ratio'])
                                # plt.scatter(x_j[:-1], Comsol_Sim_low(x_j[:-1]), c='r')

                        cost_init, cost_opt = cost_cum[:n_reg_init + (model_type != 'sogpr') * n_reg_lf_init], \
                                                  cost_cum[n_reg_init + (model_type != 'sogpr') * n_reg_lf_init:]

                        # print(len(y_hist_cum))
                        # print(len(cost_cum_high))
                        plt.plot(cost_cum_high[:-1], y_hist_cum, color=colors[model_type], label=model_type)

                        for j, y in enumerate(y_hist_init):
                            if fid_hist_init[j] == 1:
                                plt.scatter(cost_init[j], y, s=20, c='k')
                            else:
                                plt.scatter(cost_init[j], y, s=10, c='k')

                        for j, y in enumerate(y_hist_opt):
                            if fid_hist_opt[j] == 1:
                                plt.scatter(cost_opt[j], y, s=20, c=colors[model_type])
                            else:
                                plt.scatter(cost_opt[j], y, s=10, c=colors[model_type])
                        # plt.plot(cost_cum_high[1:], y_hist_cum, label=model_type, color=colors[model_type])
                        # for j, x_j in enumerate(x_hist):
                        #     if j >= 36: continue
                        #     if x_j[-1] == 1:
                        #         # plt.scatter(cost_cum_high[1:][j], Comsol_Sim_high(x_j[:-1]), c=colors[model_type])
                        #         plt.scatter(cost_cum_high[1:][j], y_hist[j], c=colors[model_type])
                        #     else:
                        #         # plt.scatter(cost_cum_high[1:][j], Comsol_Sim_low(x_j[:-1]), c=colors[model_type], alpha=.25)
                        #         plt.scatter(cost_cum_high[1:][j], y_hist[j], c=colors[model_type], alpha=.25)
plt.tight_layout()
plt.legend()
plt.show()