import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import pybenchfunction

folder_path = 'data/'

file_names = [
    # '20220309172200',
    # '20220309174208',
    # '20220314143954', # b
    # '20220331162512_1d_b_nf_False_cr_5', # badly parsed optimization cost history for mf methods.
    # '20220331164749_1d_b_nf_False_cr_25', # idem
    # '20220331165455_1d_b_nf_False_cr_45', # idem
    # '20220411202052_1d_b_nf_False_cr_5'
    '20220411202711_1d_b_nf_False_cr_25'
]

colors = {
    'sogpr': 'b',
    'cokg': 'r',
    'stmf': 'g',
}

f_class_list = pybenchfunction.get_functions(d=None, randomized_term=False)
excluded_fs = ['Ackley N. 4', 'Brown', 'Langermann', 'Michalewicz', 'Rosenbrock', 'Shubert', 'Shubert N. 3', 'Shubert N. 4']
fs = [f for f in f_class_list if f.name not in excluded_fs]
# print(fs)

for f_i, file_name in enumerate(file_names):
    open_file = open(folder_path + file_name + '.pkl', 'rb')
    data = pickle.load(open_file)
    open_file.close()

    # print(data)

    open_file = open(folder_path + file_name + '_metadata.pkl', 'rb')
    metadata = pickle.load(open_file)
    open_file.close()

    # print(metadata['dim'])

    sogpr_bests = torch.zeros((len(metadata['problem']), 1))
    cokg_bests = torch.zeros((len(metadata['problem']), 1))
    stmf_bests = torch.zeros((len(metadata['problem']), 1))
    y_opts = torch.zeros((len(metadata['problem']), 1))
    y_ranges = torch.zeros((len(metadata['problem']), 1))
    for problem_i, problem in enumerate(metadata['problem']):
        # if problem_i >= 1: continue
        # if problem_i != 2: continue
        f = fs[problem_i](d=metadata['dim'])
        x_opt, y_opt = f.get_global_minimum(d=metadata['dim'])
        # print(problem)
        x = np.linspace(f.input_domain[0][0], f.input_domain[0][1], 500)[:, None]
        y = np.apply_along_axis(f, 1, x)
        y_ranges[problem_i] = np.amax(y) - np.amin(y)
        # print(np.amax(y) - np.amin(y))

        # plt.figure(num=problem)
        print(problem)
        for model_type in metadata['model_type']:
            print(model_type)
            # if model_type != 'sogpr': continue

            y_hist_norm_agg = []
            for i in range(len(data)):
                model_type_slice = data[i][model_type]
                problem_slice = model_type_slice[problem]

                for lf in metadata['lf']:
                    print(lf)
                    lf_slice = problem_slice[lf]

                    for n_reg_init, n_reg_lf_init in zip(metadata['n_reg_init'], metadata['n_reg_lf_init']):
                        n_init = n_reg_init + (model_type != 'sogpr') * n_reg_lf_init
                        n_reg_slice = lf_slice[(n_reg_init, n_reg_lf_init)]
                        x_hist, y_hist, RAAEs, RMSTDs, opt_cost_history = n_reg_slice.values()
                        # print(y_hist)
                        #
                        # print(opt_cost_history)

                        # if model_type == 'sogpr':
                        #     plt.figure(num='RAAEs')
                        #     plt.plot((RAAEs - RAAEs[0]) / (torch.amax(RAAEs) - torch.amin(RAAEs)))
                        #
                        #     plt.figure(num='RMSTDs')
                        #     plt.plot((RMSTDs - RMSTDs[0]) / (torch.amax(RMSTDs) - torch.amin(RMSTDs)))
                        # print('RMSTDs', RMSTDs[n_init:])
                        # print(x_hist[:, -1])
                        costs = []

                        x_hist_init, x_hist_opt = x_hist[:n_init], x_hist[n_init:]
                        y_hist_init, y_hist_opt = y_hist[:n_init], y_hist[n_init:]

                        y_hist_opt_high = torch.tensor(
                            [(i, y_hist_opt[i]) for (i, _) in enumerate(x_hist_opt) if x_hist_opt[i, 1] == 1]
                        )

                        y_hist_opt_high_min = np.minimum.accumulate(y_hist_opt_high[:, 1])
                        # print(y_hist_opt_high_min)

                        y_hist_high = torch.tensor(
                            [(i, y_hist[i]) for (i, _) in enumerate(x_hist) if x_hist[i, 1] == 1]
                        )
                        y_hist_high_min = np.minimum.accumulate(y_hist_high[:, 1])
                        print(y_hist_high_min)
                        print(y_hist_opt_high_min)

                        # y_hist_min = np.minimum.accumulate(y_hist)

                        # y_hist_opt_min = y_hist_min[n_init:]
                        # y_hist_opt_min_high = torch.tensor(
                        #     [(i, y_hist_opt_min[i]) for (i, _) in enumerate(x_hist_opt) if x_hist_opt[i, 1] == 1]
                        # )

                        # print(y_hist_opt_min_high)

                        opt_cost_hist_high = torch.tensor([opt_cost_history[int(i) + 1] for i, _ in y_hist_opt_high])
                        # print(opt_cost_history, opt_cost_hist_high)
                        if problem != 'AugmentedQing':
                            y_opts[problem_i] = torch.tensor([y_opt])
                        else:
                            y_opts[problem_i] = torch.tensor([y_opt[0]])
                        if model_type == 'sogpr':
                            sogpr_bests[problem_i] = y_hist_opt_high_min[-1]
                        elif model_type == 'cokg':
                            # print(len(y_hist_opt_min_high))
                            cokg_bests[problem_i] = y_hist_opt_high_min[-1]
                        else:
                            stmf_bests[problem_i] = y_hist_opt_high_min[-1]

                        hist_vis = 1
                        if hist_vis and lf == 0.9:
                            # plt.figure(num=problem + '_' + model_type + '_' + str(lf) + '_' + str(f_i * 20 + 5))
                            # plt.plot(opt_cost_hist_high, y_hist_opt_high_min, color=colors[model_type])
                            plt.plot(opt_cost_hist_high, y_hist_high_min[-len(opt_cost_hist_high):], color=colors[model_type])
                            plt.scatter(opt_cost_hist_high, y_hist_opt_high[:, 1], c=colors[model_type])

                        # plt.plot(test[:, 1], label=model_type)
                        # plt.legend()
    # print('sogpr', sogpr_bests - y_opts)
    # print('cokg', cokg_bests - y_opts)
    # print('stmf', stmf_bests - y_opts)
    #
    # sogpr_diffs = sogpr_bests - y_opts
    # cokg_diffs = cokg_bests - y_opts
    # stmf_diffs = stmf_bests - y_opts

    for reg_f in [1e-9]:
        # print(reg_f)
        # cokg_rel = np.log10((sogpr_bests - y_opts + reg_f) / (cokg_bests - y_opts + reg_f))
        # stmf_rel = np.log10((sogpr_bests - y_opts + reg_f) / (stmf_bests - y_opts + reg_f))

        # print(sogpr_bests - y_opts)
        # print(cokg_bests - y_opts)
        # print(stmf_bests - y_opts)

        cokg_rel = np.log10(np.maximum(sogpr_bests - y_opts, reg_f) / np.maximum(cokg_bests - y_opts, reg_f))
        stmf_rel = np.log10(np.maximum(sogpr_bests - y_opts, reg_f) / np.maximum(stmf_bests - y_opts, reg_f))

        # list1 = [f.name for f in fs]
        # list2 = list(cokg_rel)
        # list3 = list(stmf_rel)
        #
        # print([(p, s) for (p, s) in zip(list1, list2)])
        # print([(p, s) for (p, s) in zip(list1, list3)])

        # print([f.name for f in fs])
        # print(cokg_rel)
        # print(stmf_rel)

        # good_indices = [i for (i, _) in enumerate(sogpr_diffs) if sogpr_diffs[i] != 0 and cokg_diffs[i] != 0 and stmf_diffs[i] != 0]
        # print(good_indices)
        #
        # rels = [[(sogpr_bests[i] - y_opts[i]) / (cokg_bests[i] - y_opts[i]),
        #                       (sogpr_bests[i] - y_opts[i]) / (cokg_bests[i] - y_opts[i])] for i in good_indices]
        # print(rels)

        # print('cokg rel', np.median(cokg_rel), [np.quantile(cokg_rel, .25), np.quantile(cokg_rel, .75)])
        # print('stmf rel', np.median(stmf_rel), [np.quantile(stmf_rel, .25), np.quantile(stmf_rel, .75)])

        # plt.figure(num='cokg_' + str(f_i * 20 + 5))
        # plt.hist(cokg_rel.numpy(), ec='k', bins=40, alpha=.5)
        # plt.figure(num='stmf_' + str(f_i * 20 + 5))
        # plt.hist(stmf_rel.numpy(), ec='k', bins=40, alpha=.5)

        # print('y range', y_ranges)
        # print('diff cokg', (sogpr_bests - cokg_bests))
        # print('diff stmf', (sogpr_bests - stmf_bests))
        # rds_cokg = np.sort(((sogpr_bests - cokg_bests) / y_ranges).flatten())
        # rds_stmf = np.sort(((sogpr_bests - stmf_bests) / y_ranges).flatten())
        # print('rel diff cokg', rds_cokg, np.median(rds_cokg), [np.quantile(rds_cokg, .25), np.quantile(rds_cokg, .75)])
        # print('rel diff stmf', rds_stmf, np.median(rds_stmf), [np.quantile(rds_stmf, .25), np.quantile(rds_stmf, .75)])
        # plt.hist(np.log(y_ranges.numpy()), ec='k')
        # plt.figure(0)
        # print('cokg', rds_cokg)
        # plt.hist(rds_cokg * 100, ec='k', bins=30, alpha=.5)
        # plt.figure(1)
        # print('stmf', rds_stmf)
        # plt.hist(rds_stmf * 100, ec='k', bins=30, alpha=.5)
plt.show()