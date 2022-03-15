import time
import pickle
import torch

from MFB.MFproblem import MFProblem
from MFB.main import bo_main
from pybenchfunction import function
from objective_formatter import botorch_TestFunction, AugmentedTestFunction

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

# fs = [function.AlpineN2, function.Ridge, function.Schwefel, function.Ackley]
fs = [function.StyblinskiTang]

dim = 1
LF = .8
cost_ratio = 10

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim), negate=True, # Minimization
            ), noise_type='bn',
        ).to(**tkwargs),
        fidelities=torch.tensor([LF, 1], **tkwargs),
        cost_ratio=cost_ratio
    )
    for f in fs
]

model_type = ['stmf']
lf = [LF]
n_reg = [2]
n_reg_lf = [2]
scramble = True
noise_fix = False
budget = 15

data_agg = []

_ = 0
while _ < 1:
    # try:
    print()
    print(_)
    print()
    data, metadata = bo_main(
        problem=problem,
        model_type=model_type,
        lf=lf,
        n_reg_init=n_reg,
        n_reg_lf_init=n_reg_lf,
        scramble=scramble,
        noise_fix=noise_fix,
        budget=budget,
    )
    data_agg.append(data)
    _ += 1
    # except:
    #     print('cokg failure')
    #     continue

metadata['dim'] = dim
metadata['cost_ratio'] = cost_ratio

folder_path = 'data/'
file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime())

open_file = open(folder_path + file_name + '.pkl', 'wb')
pickle.dump(data_agg, open_file)
open_file.close()

open_file = open(folder_path + file_name + '_metadata.pkl', 'wb')
pickle.dump(metadata, open_file)
open_file.close()

with open(folder_path + file_name + '_metadata.txt', 'w') as data:
    data.write(str(metadata))

# plt.show()