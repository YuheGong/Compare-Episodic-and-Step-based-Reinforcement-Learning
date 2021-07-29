import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

### read the file
path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
    + "cw2_log" + "/DeepMindBallInCupDMP-v2" + "/log"
df_dict = {'df{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(5)}

path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
    + "cw2_log2" + "/DeepMindBallInCupDenseDMP-v2" + "/log"
ddf_dict = {'ddf{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(5)}

df_mean = {}
ddf_mean = {}

df_mean["success_rate_full"] = (df_dict["df1"]["success_rate_full"] +df_dict["df2"]["success_rate_full"] + df_dict["df3"]["success_rate_full"] + df_dict["df4"]["success_rate_full"]) / 5
ddf_mean["success_rate_full"] = (ddf_dict["ddf1"]["success_rate_full"] + ddf_dict["ddf2"]["success_rate_full"] + ddf_dict["ddf3"]["success_rate_full"] + ddf_dict["ddf4"]["success_rate_full"]) /5
#print(df_mean)

df_mean["total_samples"]  = df_dict["df1"]["total_samples"] * 250
ddf_mean["total_samples"]  = ddf_dict["ddf1"]["total_samples"] * 250


X_Y_Spline = make_interp_spline(df_mean["total_samples"], df_mean["success_rate_full"])
X1 = np.linspace(df_mean["total_samples"].min(), df_mean["total_samples"].max(), 100)
Y1 = X_Y_Spline(X1)
plt.plot(X1, Y1, label="DMP One step log reward")


X_Y_Spline = make_interp_spline(ddf_mean["total_samples"], ddf_mean["success_rate_full"])
X2 = np.linspace(ddf_mean["total_samples"].min(), ddf_mean["total_samples"].max(), 100)
Y2 = X_Y_Spline(X1)
plt.plot(X2, Y2, label="DMP Full Step log reward")


plt.title('success rate according to samples')
plt.legend()
plt.show()
