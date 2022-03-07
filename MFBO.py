import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
import GPy
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from GPy.models.gp_regression import GPRegression
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.acquisition import Acquisition
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
import os
import time
import matplotlib.pyplot as plt
import matplotlib

start_time = time.time()
low_fid_time = 0
high_fid_time = 0
folder_path = os.getcwd()

np.random.seed(12345)
n_fidelities = 2
N_Design_param = 9
low_fidelity_cost = 1
high_fidelity_cost = 10
N_Low_Fid_Init = 50
N_High_Fid_Init = 25
Maximum_Iter_Time = 14 * 24 * 3600

parameter_space = ParameterSpace([ContinuousParameter('x0', 0, 1),
                                  ContinuousParameter('x1', 0, 1),
                                  ContinuousParameter('x2', 0, 1),
                                  ContinuousParameter('x3', 0, 1),
                                  ContinuousParameter('x4', 0, 1),
                                  ContinuousParameter('x5', 0, 1),
                                  ContinuousParameter('x6', 0, 1),
                                  ContinuousParameter('x7', 0, 1),
                                  ContinuousParameter('x8', 0, 1),
                                  ContinuousParameter('x9', 0, 1),
                                  InformationSourceParameter(n_fidelities)])

# These files must exist; provide your own paths
a_Design_var1_txt_file = "{}/a1_Design_var1.txt".format(folder_path)
a_Design_var2_txt_file = "{}/a1_Design_var2.txt".format(folder_path)
a_Design_var3_txt_file = "{}/a1_Design_var3.txt".format(folder_path)
a_Design_var4_txt_file = "{}/a1_Design_var4.txt".format(folder_path)
a_Design_var5_txt_file = "{}/a1_Design_var5.txt".format(folder_path)
a_Design_var6_txt_file = "{}/a1_Design_var6.txt".format(folder_path)
a_Design_var7_txt_file = "{}/a1_Design_var7.txt".format(folder_path)
a_Design_var8_txt_file = "{}/a1_Design_var8.txt".format(folder_path)
a_Design_var9_txt_file = "{}/a1_Design_var9.txt".format(folder_path)
b_Objective_1_txt_file = "{}/a2_Objective_1.txt".format(folder_path)
b_Simu_Time_1_txt_file = "{}/a2_Simu_Time_1.txt".format(folder_path)

# Command line to run comsol; provide your own file name & check problem file itself
cmdl_lf = "cd {}/ && comsol mphserver matlab a0_Validation_MFBO_LF -nodesktop -mlnosplash".format(folder_path)
cmdl_hf = "cd {}/ && comsol mphserver matlab a0_Validation_MFBO_HF -nodesktop -mlnosplash".format(folder_path)


def Comsol_Sim():
    user_function = MultiSourceFunctionWrapper([
        lambda x: Comsol_Sim_low(x),
        lambda x: Comsol_Sim_high(x)])
    return user_function


# def Comsol_Sim_high(x):
#     open(a_Design_var1_txt_file, "w").write(str(x[0]))
#     open(a_Design_var2_txt_file, "w").write(str(x[1]))
#     open(a_Design_var3_txt_file, "w").write(str(x[2]))
#     open(a_Design_var4_txt_file, "w").write(str(x[3]))
#     open(a_Design_var5_txt_file, "w").write(str(x[4]))
#     open(a_Design_var6_txt_file, "w").write(str(x[5]))
#     open(a_Design_var7_txt_file, "w").write(str(x[6]))
#     open(a_Design_var8_txt_file, "w").write(str(x[7]))
#     open(a_Design_var9_txt_file, "w").write(str(x[8]))
#     os.system(cmdl_hf)
#     os.chdir(folder_path)
#     return float(open(b_Objective_1_txt_file, "r").read().strip())
#
#
# def Comsol_Sim_low(x):
#     open(a_Design_var1_txt_file, "w").write(str(x[0]))
#     open(a_Design_var2_txt_file, "w").write(str(x[1]))
#     open(a_Design_var3_txt_file, "w").write(str(x[2]))
#     open(a_Design_var4_txt_file, "w").write(str(x[3]))
#     open(a_Design_var5_txt_file, "w").write(str(x[4]))
#     open(a_Design_var6_txt_file, "w").write(str(x[5]))
#     open(a_Design_var7_txt_file, "w").write(str(x[6]))
#     open(a_Design_var8_txt_file, "w").write(str(x[7]))
#     open(a_Design_var9_txt_file, "w").write(str(x[8]))
#     os.system(cmdl_lf)
#     os.chdir(folder_path)
#     return float(open(b_Objective_1_txt_file, "r").read().strip())

# def Comsol_Sim_high(x):
#     print("HIGH")
#     return x[0] ** 2 + x[1] ** 2
#     # return np.sin(x)
#
#
# def Comsol_Sim_low(x):
#     # return x[0] ** 2 + x[1] ** 2
#     print("LOW")
#     return (x[0] ** 2 + x[1] ** 2) / 2 + x[0]
#     # return np.sin(x + 1) + .1 * x

def Comsol_Sim_high(x):
    print("HIGH")
    x0=x[0]*10-5
    x1=x[1]*10-5
    return ((x0 ** 4 - 16 * x0 ** 2 + 5 * x0 )/2 + (x1 ** 4 - 16 * x1 ** 2 + 5 * x1 )/2)


def Comsol_Sim_low(x):
    print("LOW")
    x0=x[0]*10-5
    x1=x[1]*10-5
    return ((x0 ** 4 - 16 * x0 ** 2 + 5 * x0 )/2 + (x1 ** 4 - 16 * x1 ** 2 + 5 * x1 )/2)*0.1+x0-x1


class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evalute(x), np.zeros(x.shape)


# Objective_Function = Comsol_Sim()
# Objective_Function_low = Objective_Function.f[0]
# Objective_Function_high = Objective_Function.f[1]
#
# x_random = np.random.rand(max([N_Low_Fid_Init, N_High_Fid_Init]) * N_Design_param)[:, None]
#
# x_low = np.reshape(x_random[0:N_Low_Fid_Init * N_Design_param, :], (N_Low_Fid_Init, -1))
# x_high = np.reshape(x_random[0:N_High_Fid_Init * N_Design_param, :], (N_High_Fid_Init, -1))
#
# start_time_iter = time.time()
# X_init_low = x_low[0]
# Y_init_low = Objective_Function_low(X_init_low)
# low_fid_time = low_fid_time + time.time() - start_time_iter
# open(b_Simu_Time_1_txt_file, "a").write('LF' + str(1) + ':' + str(time.time() - start_time) + '\n')
#
# Low_Fid_iter = 1
# while Low_Fid_iter < N_Low_Fid_Init:
#     start_time_iter = time.time()
#
#     x_low_next = x_low[Low_Fid_iter]
#     y_low_next = Objective_Function_low(x_low_next)
#
#     X_init_low = np.vstack((X_init_low, x_low_next))  # Combining older steps with new step
#     Y_init_low = np.vstack((Y_init_low, y_low_next))
#
#     Low_Fid_iter += 1
#     low_fid_time = low_fid_time + time.time() - start_time_iter
#
#     print("low fidelity iteration ", Low_Fid_iter)
#     print("time :", time.time() - start_time)
#     print("input :", x_low_next)
#     print("output :", y_low_next)
#     open(b_Simu_Time_1_txt_file, "a").write('LF' + str(Low_Fid_iter) + ':' + str(time.time() - start_time) + '\n')
#
# start_time_iter = time.time()
# X_init_high = x_high[0]
# Y_init_high = Objective_Function_high(X_init_high)
# high_fid_time = high_fid_time + time.time() - start_time_iter
# open(b_Simu_Time_1_txt_file, "a").write('HF' + str(1) + ':' + str(time.time() - start_time) + '\n')
#
# High_Fid_iter = 1
# while High_Fid_iter < N_High_Fid_Init:
#     start_time_iter = time.time()
#
#     x_high_next = x_high[High_Fid_iter]
#     y_high_next = Objective_Function_high(x_high_next)
#
#     X_init_high = np.vstack((X_init_high, x_high_next))  # Combining older steps with new step
#     Y_init_high = np.vstack((Y_init_high, y_high_next))
#
#     High_Fid_iter += 1
#     high_fid_time = high_fid_time + time.time() - start_time_iter
#
#     print("high fidelity iteration ", High_Fid_iter)
#     print("time :", time.time() - start_time)
#     print("input :", x_high_next)
#     print("output :", y_high_next)
#     open(b_Simu_Time_1_txt_file, "a").write('HF' + str(High_Fid_iter) + ':' + str(time.time() - start_time) + '\n')
#
# kern_low = GPy.kern.RBF(input_dim=N_Design_param, variance=1)
# kern_low.lengthscale.constrain_bounded(0.01, 0.5)
#
# kern_err = GPy.kern.RBF(input_dim=N_Design_param, variance=1)
# kern_err.lengthscale.constrain_bounded(0.01, 0.5)
#
# multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
#
# while time.time() - start_time <= Maximum_Iter_Time:
#
#     start_time_iter = time.time()
#
#     x_array, y_array = convert_xy_lists_to_arrays([X_init_low, X_init_high], [Y_init_low, Y_init_high])
#     gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, n_fidelities)
#
#     gpy_model.likelihood.Gaussian_noise.fix(0.1)
#     gpy_model.likelihood.Gaussian_noise_1.fix(0.1)
#
#     model = GPyMultiOutputWrapper(gpy_model, n_fidelities, 5, verbose_optimization=False)
#     model.optimize()
#
#     # low_fidelity_cost  = low_fid_time/Low_Fid_iter
#     # high_fidelity_cost = high_fid_time/High_Fid_iter
#
#     cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
#     acquisition = MultiInformationSourceEntropySearch(model, parameter_space) / cost_acquisition
#
#     initial_loop_state = create_loop_state(x_array, y_array)
#     acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(parameter_space),
#                                                             parameter_space)
#     candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
#     model_updater = FixedIntervalUpdater(model)
#     loop = OuterLoop(candidate_point_calculator, model_updater, initial_loop_state)
#     new_x = loop.candidate_point_calculator.compute_next_points(Objective_Function)
#
#     if new_x[:, -1] == 0.:
#         start_time_iter = time.time()
#         x_low_next = np.reshape(new_x[:, 0:N_Design_param], (-1))
#         y_low_next = Objective_Function_low(x_low_next)
#         X_init_low = np.vstack((X_init_low, x_low_next))  # Combining older steps with new step
#         Y_init_low = np.vstack((Y_init_low, y_low_next))
#         Low_Fid_iter += 1
#         low_fid_time = low_fid_time + time.time() - start_time_iter
#         open(b_Simu_Time_1_txt_file, "a").write('LF' + str(Low_Fid_iter) + ':' + str(time.time() - start_time) + '\n')
#
#     elif new_x[:, -1] == 1.:
#         start_time_iter = time.time()
#         x_high_next = np.reshape(new_x[:, 0:N_Design_param], (-1))
#         y_high_next = Objective_Function_high(x_high_next)
#         X_init_high = np.vstack((X_init_high, x_high_next))  # Combining older steps with new step
#         Y_init_high = np.vstack((Y_init_high, y_high_next))
#         High_Fid_iter += 1
#         high_fid_time = high_fid_time + time.time() - start_time_iter
#         open(b_Simu_Time_1_txt_file, "a").write('HF' + str(High_Fid_iter) + ':' + str(time.time() - start_time) + '\n')
#
#     else:
#         print("Fidelity Error")
#
#     print("low fidelity iteration ", Low_Fid_iter)
#     print("High fidelity iteration ", High_Fid_iter)
#     print("time :", time.time() - start_time_iter)
#     print("input :", x_high_next)
#     print("output :", y_high_next)
