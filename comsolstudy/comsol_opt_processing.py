import pickle

import matplotlib.pyplot as plt

from MFB.MFBO import Comsol_Sim_high

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
    '20220303171503', # sogpr
    '20220303171823', # stmf
    '20220303171947', # cokg

    # # 2D Styblinsky-Tang
    # '20220303172116', # sogpr
    # '20220303172321', # stmf
    # '20220303172447', # cokg
]

for f_i, file_name in enumerate(file_names):
    print(file_name)
    open_file = open(folder_path + file_name + '.pkl', 'rb')
    data = pickle.load(open_file)
    open_file.close()

    # print(data)

    open_file = open(folder_path + file_name + '_metadata.pkl', 'rb')
    metadata = pickle.load(open_file)
    open_file.close()

    # print(metadata['budget'])

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
                        x_hist_init, x_hist_opt = x_hist[:n_reg_init + n_reg_lf_init], x_hist[n_reg_init + n_reg_lf_init:]
                        print(x_hist)

                        # print(np.sum(x_hist_opt[:, -1] == 1))
                        y_hist_cum = []
                        cost_cum = [0]
                        cost_cum_high = [0]
                        y_min_inc = 1e20 # large initialization value

                        for j, x_j in enumerate(x_hist):
                            if x_j[-1] == 1:
                                y_min_inc = min(y_min_inc, Comsol_Sim_high(x_j[:-1]))
                                y_hist_cum.append(y_min_inc)
                                # plt.scatter(x_j[:-1], Comsol_Sim_high(x_j[:-1]), c='g')
                                cost_cum.append(cost_cum[-1] + 1)
                                cost_cum_high.append(cost_cum[-1])
                            else:
                                cost_cum.append(cost_cum[-1] + .1)
                                # plt.scatter(x_j[:-1], Comsol_Sim_low(x_j[:-1]), c='r')
                        plt.plot(cost_cum_high[1:], y_hist_cum, label=model_type)

plt.tight_layout()
plt.legend()
plt.show()
