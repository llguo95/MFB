import pickle

import matplotlib.pyplot as plt
import numpy as np

folder_path = 'data/'

file_names = [
    '20220211165001',
    '20220211170131'
]

mrmv = []
for f_i, file_name in enumerate(file_names):
    open_file = open(folder_path + file_name + '.pkl', 'rb')
    data = pickle.load(open_file)
    open_file.close()

    # print(data)

    open_file = open(folder_path + file_name + '_metadata.pkl', 'rb')
    metadata = pickle.load(open_file)
    open_file.close()

    # print(metadata['budget'])

    mrm = np.zeros((29, len(metadata['model_type'])))

    for model_no, model_type in enumerate(metadata['model_type']):
        model_type_slice = data[model_type]

        # print(model_type)

        for problem_no, problem in enumerate(metadata['problem']):
            problem_slice = model_type_slice[problem]

            # print(problem)

            for lf in metadata['lf'][2:3]:
                lf_slice = problem_slice[lf]

                # print(lf)

                for n_reg, n_reg_lf in zip(metadata['n_reg'], metadata['n_reg_lf']):
                    n_reg_slice = lf_slice[(n_reg, n_reg_lf)]

                    # print(n_reg_slice['RAAE_stats']['median'])
                    mrm[problem_no, model_no] = n_reg_slice['RAAE_stats']['median']
                    # print(n_reg_slice['RAAE_stats']['quantiles'])

    mrmv.append(mrm)

for i in range(29):
    mrmv[0][i] = mrmv[1][i] / mrmv[0][i]

k = 2
print(mrmv[0][:, k])
plt.hist(np.log10(mrmv[0][:, k]), bins=20)
plt.show()