import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

def dict_building(folder, name):
    #print("name", name)
    path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
           + "logs/" + folder + "/" + name
    plotdict = {'df{}'.format(q): pd.read_csv(path + "_{}".format(q) + "/data.csv", nrows=2000) for q in
                range(1,2)}
    #plotdict.update({'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
    #                 range(10, 20)})
    return plotdict


def plot_function(success_rate_value, plotdict, name, samples):

    plot_mean = np.mean(success_rate_value,axis=0)
    #print("success_rate", success_rate_value)
    plot_std = np.sort(np.std(success_rate_value, axis=0))
    plot_var = np.zeros(len(success_rate_value[0]))
    for i in range(len(success_rate_value[0])):
        plot_var[i] = np.std(success_rate_value[:,i])
    #print("std", plot_std)
    #print("(success_rate_value",success_rate_value[:,74])
    #print(plot_var)
    #print("plotdict", plotdict)
    plot_samples = plotdict["df1"]["Unnamed: 0"] * samples

    #print(plot_samples)
    #print(np.zeros(2000).shape)
    #X_Y_Spline = make_interp_spline(plot_mean, plot_samples)
    #X = np.linspace(plot_mean.min(), plot_mean.max(), 100)
    #Y = X_Y_Spline(X)
    #print(plot_mean-plot_var)
    #print(plot_mean+plot_var)
    #assert 1==239
    plt.plot(plot_samples, plot_mean, label=name)#label_name(name))
    plt.fill_between(plot_samples, plot_mean-plot_var, plot_mean+plot_var, alpha=0.1)
    return plotdict


def suc_rate_value(plotdict, value):
    success_rate_full = []
    #print("plotdict.items()",plotdict.items())
    for k in plotdict.items():
        success_rate_full.append(k[1][value])
    success_rate_value = np.array(success_rate_full)
    return success_rate_value

def label_name(name):
    if "ppo" in name:
        a = "DMP"
    elif "ProMP" in name:
        a = "ProMP"
    if name[-1] == "0":
        b = 'exp'
    elif name[-1] == "1":
        b = 'quad'
    elif name[-1] == "2":
        b = 'log'
    if "Dense" in name:
        c = "dense"
    else:
        c = "sparse"
    return a + ' - ' + b + ' - ' + c


#folder = "ppo"
#samples = 20000

folder = "sac"
samples = 1000
value = "eval/mean_reward"


#folder = "forthweek"
#value = "reward"

for v in range(2,3):
    #name = "DeepMindBallInCup" + algo + "-v{}".format(v)
    name = "ALRReacherBalanceSparse" + "-v{}".format(v)
    plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict,value)
    name = "sac - sparse"
    plot_function(success_rate_value, plotdict, name, samples)

    folder = "sac"
    samples = 1000
    value = "eval/mean_reward"
    name = "ALRReacherBalance" + "-v{}".format(v)
    plotdict = dict_building(folder, name)
    success_rate_value = np.array(suc_rate_value(plotdict, value))
    name = "sac - dense"
    plot_function(success_rate_value, plotdict, name, samples)

    value = "iteration/reward"
    folder = "promp"
    samples = 2000
    name = "ALRReacherBalanceSparseProMP" + "-v{}".format(v)
    plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict, value)
    name = "promp - sparse"
    plot_function(success_rate_value, plotdict, name, samples)

    value = "iteration/reward"
    folder = "promp"
    samples = 2000
    name = "ALRReacherBalanceProMP" + "-v{}".format(v)
    plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict, value)
    name = "promp - dense"
    plot_function(success_rate_value, plotdict, name, samples)



plt.title( 'ALRReacher rewards')
plt.legend()
plt.show()


