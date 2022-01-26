import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

data = {}
data["path_in"] = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
           + "logs/ppo/" + "DeepMindBallInCup-v2_1/PPO_1"
data["path_out"] = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning" \
           + "/logs/data.csv"


def dict_building(folder, name):
    #print("name", name)
    path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
           + "logs/" + folder + "/" + name
    plotdict = {'df{}'.format(q): pd.read_csv(path + "_{}".format(q) + "/data.csv", nrows=2000) for q in
                range(1,2)}
    #plotdict.update({'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
    #                 range(10, 20)})
    return plotdict


def plot_function(success_rate_value, plotdict, name, samples, alpha=1, zorder = 1):

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
    #print("plot_mean",plot_samples)
    #X_Y_Spline = make_interp_spline(plot_mean, plot_samples)
    #plot_mean = np.linspace(plot_mean.min(), plot_mean.max(), 100)
    #plot_samples = X_Y_Spline(X)
    #print(plot_mean-plot_var)
    #print(plot_mean+plot_var)
    #assert 1==239
    plt.plot(plot_samples, plot_mean, label=label_name(name), alpha=alpha, zorder=zorder)#label_name(name))
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
    print(name)
    if "DMP" in name:
        a = "DMP"
    elif "ProMP" in name:
        a = "ProMP"
    elif "ppo" in name:
        a = "PPO"
    elif "sac" in name:
        a = "SAC"
    if name[-1] == "0":
        b = 'Random Position & Random Velocity'
    elif name[-1] == "1":
        b = 'Same Position & Same Velocity'
    elif name[-1] == "2":
        b = 'Fixed Position & No Velocity'
    if "Dense" in name:
        c = 'Fix Goal&No Velocity'
    else:
        c = "sparse"
    return a + ' - ' + b


folder = "ppo"
samples = 20000

folder = "sac"
samples = 500 * 10

value = "eval/mean_reward"
version = 1

if version == 1:
    beta = 0.5
else: beta = 1
#folder = "forthweek"
#value = "reward"

for v in range(version-1, version):
    folder = "ppo"
    samples = 100000 * 5 / 3

    # name = "DeepMindBallInCupDense" + algo + "-v{}".format(v)
    name = "ALRReacherBalance" + "-v{}".format(v)
    plotdict = dict_building(folder, name)
    success_rate_value = np.array(suc_rate_value(plotdict, value))
    name = folder + name
    plot_function(success_rate_value, plotdict, name, samples, zorder = 10)

    print("beta", beta)
    folder = "sac"
    samples = 500 * 10 * beta
    #name = "DeepMindBallInCup" + algo + "-v{}".format(v)
    name = "ALRReacherBalance" + "-v{}".format(v)
    #plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict,value)
    name = folder + name
    plot_function(success_rate_value, plotdict, name, samples,zorder=3)
    #assert 1==239

    value = "iteration/reward"
    folder = "dmp"
    samples = 500 * 10 * beta
    # name = "DeepMindBallInCup" + algo + "-v{}".format(v)
    name = "ALRReacherBalanceDMP" + "-v{}".format(v)
    # plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict, value)
    name = folder + name
    plot_function(success_rate_value, plotdict, name, samples, zorder=2)
    # assert 1==239

    value = "iteration/reward"
    folder = "promp"
    samples = 500 * 10 * beta
    # name = "DeepMindBallInCup" + algo + "-v{}".format(v)
    name = "ALRReacherBalanceProMP" + "-v{}".format(v)
    # plotdict = {}
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict, value)
    name = folder + name
    plot_function(success_rate_value, plotdict, name, samples, zorder=2)


    #plot_function(folder, "DeepMindBallInCupProMP-v{}".format(v))
    #plot_function(folder, "DeepMindBallInCupDenseProMP-v{}".format(v))


plt.title(version)
plt.legend()#bbox_to_anchor=(1,0), loc=3, borderaxespad=0)
plt.show()


