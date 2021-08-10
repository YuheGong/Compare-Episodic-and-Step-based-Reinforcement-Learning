import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

### read the file
path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
    + "DMP_ohne_act" + "/DeepMindBallInCupDMP-v0" + "/log"
prodf_dict = {'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(20)}

#path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
#    + "secondweek/ProMP" + "/DeepMindBallInCupDenseProMP-v0" + "/log"
#proddf_dict = {'ddf{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(5)}



#ath = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
#    + "secondweek" + "/DeepMindBallInCupDMP-v0" + "/log"
#df_dict = {'df{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(5)}

#path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/" \
#    + "secondweek" + "/DeepMindBallInCupDenseDMP-v0" + "/log"
#ddf_dict = {'ddf{}'.format(q): pd.read_csv(path + "/rep_0{}".format(q) + "/rep_{}".format(q) + ".csv") for q in range(5)}

#f_mean = {}
#ddf_mean = {}
prodf_mean = {}
#proddf_mean = {}
prodf_mean["success_rate_full"] = prodf_dict["df1"]["success_rate_full"]

#print("dfkeys", df_dict)
for i in range(1,20):
    prodf_mean["success_rate_full"] += prodf_dict["df{}".format(i)]["success_rate_full"]

prodf_mean["success_rate_full"] = prodf_mean["success_rate_full"]/20



#ddf_mean["success_rate_full"] = (ddf_dict["ddf0"]["success_rate_full"]+ ddf_dict["ddf1"]["success_rate_full"] + ddf_dict["ddf2"]["success_rate_full"] + ddf_dict["ddf3"]["success_rate_full"] + ddf_dict["ddf4"]["success_rate_full"]) /5
#print(df_mean)
#prodf_mean["success_rate_full"]= (prodf_dict["df0"]["success_rate_full"]+ prodf_dict["df1"]["success_rate_full"] +prodf_dict["df2"]["success_rate_full"] + prodf_dict["df3"]["success_rate_full"] + prodf_dict["df4"]["success_rate_full"]) / 5
#proddf_mean["success_rate_full"] = (proddf_dict["ddf0"]["success_rate_full"]+ proddf_dict["ddf1"]["success_rate_full"] + proddf_dict["ddf2"]["success_rate_full"] + proddf_dict["ddf3"]["success_rate_full"] + proddf_dict["ddf4"]["success_rate_full"]) /5

#df_mean["total_samples"]  = df_dict["df1"]["total_samples"] * 250
#ddf_mean["total_samples"]  = ddf_dict["ddf1"]["total_samples"] * 250
prodf_mean["total_samples"]  = prodf_dict["df1"]["total_samples"] * 50
#proddf_mean["total_samples"]  = proddf_dict["ddf1"]["total_samples"] * 250

'''
X_Y_Spline = make_interp_spline(df_mean["total_samples"], df_mean["success_rate_full"])
X1 = np.linspace(df_mean["total_samples"].min(), df_mean["total_samples"].max(), 100)
Y1 = X_Y_Spline(X1)
plt.plot(X1, Y1, label = "DMP exp sparse reward") #label="DMP One step log success_rate_full")

print('ddf_mean["total_samples"]',ddf_mean["total_samples"])
print('ddf_mean["success_rate_full"]',ddf_mean["success_rate_full"])
X_Y_Spline = make_interp_spline(ddf_mean["total_samples"], ddf_mean["success_rate_full"])
X2 = np.linspace(ddf_mean["total_samples"].min(), ddf_mean["total_samples"].max(), 100)
Y2 = X_Y_Spline(X1)
plt.plot(X2, Y2, label="DMP exp dense reward")
'''
X_Y_Spline = make_interp_spline(prodf_mean["total_samples"], prodf_mean["success_rate_full"])
X3 = np.linspace(prodf_mean["total_samples"].min(), prodf_mean["total_samples"].max(), 100)
Y3 = X_Y_Spline(X3)
plt.plot(X3, Y3, label = "ProMP exp sparse reward") #label="DMP One step log success_rate_full")
'''
X_Y_Spline = make_interp_spline(proddf_mean["total_samples"], proddf_mean["success_rate_full"])
X4 = np.linspace(proddf_mean["total_samples"].min(), proddf_mean["total_samples"].max(), 100)
Y4 = X_Y_Spline(X4)
plt.plot(X4, Y4, label = "ProMP exp dense reward") #label="DMP One step log success_rate_full")
'''
plt.title('success rate according to samples')
plt.legend()
plt.show()



