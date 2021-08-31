import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

### read the file
path = "/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/"

sparse = pd.read_csv(path + "test.csv")
#a = [row[0] for row in sparse]+
a = sparse["Unnamed: 0"] * 50
#print("success_rate", int(sparse["success_rate"].values[0][1]))
sparse["b"] = [int(sparse["success_rate"].values[i][1]) for i in range(len(sparse['success_rate']))]

i = 0
print(sparse['success_rate'].index[-1])
exceed = False
while i < sparse['success_rate'].index[-1]:
    #m = 0
    c = 0
    for d in range(10):
        if sparse["b"].values[d+i]:

        #print(d, i)
            c += sparse["b"].values[d+i]
        else:
            exceed = True
            print("excceed")
    if exceed:
        break
    c = c / 10
    for d in range(10):
        sparse["b"].values[i+d] = c
    i += 10
    if i > sparse['success_rate'].index[-1]-9:
        break
    #print(c)
    #while m < 10:

print(sparse['b'].values)#[1:30])
#X_Y_Spline = make_interp_spline(a, sparse["b"])
#X3 = np.linspace(a.min(), a.max(), 100)
#Y3 = X_Y_Spline(X3)
#plt.plot(X3, Y3, label="DeepMindBallInCupDMP-v0") #label="DMP One step log success_rate_full")

#plt.title('success rate according to samples')
#plt.legend()
#plt.show()




