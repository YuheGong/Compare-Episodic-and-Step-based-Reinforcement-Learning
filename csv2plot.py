import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt


def dict_building(folder,name):
    path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
           + folder + "/" + name + "/log"
    plotdict = {'df{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
                range(10)}
    plotdict.update({'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
                     range(10, 20)})
    return plotdict


def plot_function(success_rate_value, plotdict, name):

    plot_mean = np.mean(success_rate_value,axis=0)
    plot_std = np.sort(np.std(success_rate_value, axis=0))
    plot_samples = plotdict["df0"]["total_samples"] * 50

    #print(plot_samples)
    #print(np.zeros(2000).shape)
    #X_Y_Spline = make_interp_spline(plot_mean, plot_samples)
    #X = np.linspace(plot_mean.min(), plot_mean.max(), 100)
    #Y = X_Y_Spline(X)
    plt.plot(plot_samples, plot_mean, label=name)
    plt.fill_between(plot_samples, plot_mean-plot_std, plot_mean+plot_std, alpha=0.1)
    return plotdict


def suc_rate_value(plotdict, value):
    success_rate_full = []
    for k in plotdict.items():
        success_rate_full.append(k[1][value])
    success_rate_value = np.array(success_rate_full)
    return success_rate_value

folder = "forthweek"
value = "reward"

for v in range(3):
    name = "DeepMindBallInCupDMP-v{}".format(v)
    #name = "DeepMindBallInCupProMP-v{}".format(v)
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict,value)
    plot_function(success_rate_value, plotdict, name)

    name = "DeepMindBallInCupDenseDMP-v{}".format(v)
    #name = "DeepMindBallInCupDenseProMP-v{}".format(v)
    plotdict = dict_building(folder, name)
    success_rate_value = suc_rate_value(plotdict,value)
    plot_function(success_rate_value, plotdict, name)

    #plot_function(folder, "DeepMindBallInCupProMP-v{}".format(v))
    #plot_function(folder, "DeepMindBallInCupDenseProMP-v{}".format(v))


plt.title('success rate according to samples')
plt.legend()
plt.show()



