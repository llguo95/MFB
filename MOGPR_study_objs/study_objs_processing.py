import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder_path = 'data/'

file_names = [
    # dim 1
    # '20220211165001', # mf, 30LF b
    # '20220218110629', # mf, 5LF b
    # '20220218144547', # mf, 30LF b, matern test
    # '20220218172924', # mf, 5LF bn
    # '20220218180630', # mf, 30LF bn
    # '20220222132448', # mf, 30LF bn, matern test
    # '20220222174825', # mf, 15LF bn
    '20220222193253', # mf, 15LF b

    # dim 2
    # '20220222143107', # mf, 75LF bn
    # '20220222185639', # mf, 75LF b

    # dim 1
    '20220211170131', # sogpr, 6HF
    # '20220221174116', # sogpr, 7HF
    # '20220221174326', # sogpr, 8HF

    # dim 2
    # '20220222144503' # sogpr, 30HF
]

model_types = ['cokg', 'cokg_dms', 'mtask']
lfs = [0.1, 0.5, 0.9]

mrmv = []
metadata_mf = None
for f_i, file_name in enumerate(file_names):
    open_file = open(folder_path + file_name + '.pkl', 'rb')
    data = pickle.load(open_file)
    open_file.close()

    # print(data)

    open_file = open(folder_path + file_name + '_metadata.pkl', 'rb')
    metadata = pickle.load(open_file)
    if f_i == 0:
        metadata_mf = metadata
    open_file.close()

    # print(metadata['budget'])

    # mrm = np.zeros((29, len(metadata['model_type'])))
    mrm_model_type = {}
    for model_no, model_type in enumerate(metadata['model_type']):
        model_type_slice = data[model_type]

        # print(model_type)
        mrm_problem = {}
        for problem_no, problem in enumerate(metadata['problem']):
            if problem == 'AugmentedRidge': continue
            problem_slice = model_type_slice[problem]

            # print(problem)

            mrm_lf = {}
            for lf in metadata['lf']:
                lf_slice = problem_slice[lf]

                # print(lf)

                mrm_n_reg = []
                for n_reg, n_reg_lf in zip(metadata['n_reg'], metadata['n_reg_lf']):
                    n_reg_slice = lf_slice[(n_reg, n_reg_lf)]

                    # print(n_reg_slice['RAAE_stats']['median'])
                    # mrm[problem_no, model_no] = n_reg_slice['RAAE_stats']['median']
                    mrm_n_reg.append(n_reg_slice['RAAE_stats']['median'])

                    # print(n_reg_slice['RAAE_stats']['quantiles'])
                mrm_lf[lf] = mrm_n_reg
            mrm_problem[problem] = mrm_lf
        mrm_model_type[model_type] = mrm_problem
    mrmv.append(mrm_model_type)

mrmv_mf, mrmv_sf = mrmv[:-1], mrmv[-1]

# print(mrmv_mf)

for mf_i, mf_scenario in enumerate(mrmv_mf):
    df_data = {}
    for model_type in mf_scenario:
        fig, axs = plt.subplots(num=model_type + str(mf_i), ncols=1, nrows=3, sharex=True, sharey=True, figsize=(5, 8))
        for lf_i, lf in enumerate(metadata_mf['lf']):
            prob_stat_vec = []
            for problem in metadata_mf['problem']:
                if problem == 'AugmentedRidge': continue
                # mrmv_mf[model_type][problem][lf][0] = mrmv_sf['sogpr'][problem][0.1][0] / mrmv_mf[model_type][problem][lf][0]
                # print(problem, mrmv_sf['sogpr'][problem][0.1][0], mrmv_mf[model_type][problem][lf][0])
                lri = np.log10(mrmv_sf['sogpr'][problem][0.5][0] / mf_scenario[model_type][problem][lf][0])
                # lri = np.log10(mrmv_sf['sogpr'][problem][0.1][0] / mf_scenario[model_type][problem][lf][0])
                # lri = mrmv_sf['sogpr'][problem][0.1][0] / mrmv_mf[model_type][problem][lf][0]
                prob_stat_vec.append(lri)

            df_data[lf] = prob_stat_vec
            print()
            # print(model_type, lf, 10 ** np.median(prob_stat_vec), [10 ** np.quantile(prob_stat_vec, q=.25), 10 ** np.quantile(prob_stat_vec, q=.75)])
            print(model_type, lf, np.median(prob_stat_vec), [np.quantile(prob_stat_vec, q=.25), np.quantile(prob_stat_vec, q=.75)])
            ax = axs[lf_i]
            ax.hist(prob_stat_vec, bins=20, ec='k')
            ax.set_title('LF = ' + str(lf))
            ax.axvline(x=0, color='k', linestyle='--', linewidth=3)
            if lf_i == 2:
                ax.set_xlabel('Relative improvement (orders of mag.)')
            ax.set_ylabel('Frequency')
            # ax.set_xscale('log')
            # axs[lf_i].grid()
        plt.tight_layout()

        # df = pd.DataFrame(data=df_data, index=metadata_mf['problem'])
        # print(df)

        # df.to_excel('data/' + file_names[0] + '_data_' + model_type + '.xlsx')

plt.show()