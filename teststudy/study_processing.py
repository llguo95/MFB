import pickle

import matplotlib.pyplot as plt
import numpy as np

folder_path = 'data/'

file_names = [
    '20220224134858'
]

for f_i, file_name in enumerate(file_names):
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
                        y_hist = np.minimum.accumulate(lf_slice[(n_reg_init, n_reg_lf_init)]['y_hist_norm'][:, 0])
                        n_reg_slice = lf_slice[(n_reg_init, n_reg_lf_init)]

                        print(n_reg_slice)
                        # print(y_hist)
