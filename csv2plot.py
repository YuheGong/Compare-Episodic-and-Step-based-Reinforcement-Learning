import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt



df_1 = pd.read_csv('/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/rosenbrock/log/rep_00/rep_0.csv')
df_2 = pd.read_csv('/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/rosenbrock/log/rep_00/rep_1.csv')
#df_3 = pd.read_csv('/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/rosenbrock/log/rep_00/rep_2.csv')
#df_4 = pd.read_csv('/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/rosenbrock/log/rep_00/rep_0.csv')


#print(df["total_samples"])
#fig = px.line(df, x = "total_samples", y = "reward", title='Apple Share Prices over time (2014)')
#fig.show()

# create data
#x = [10, 20, 30, 40, 50]
#y = [30, 30, 30, 30, 30]

# plot lines
plt.plot(df_1["total_samples"], df_1["reward"], label="line 1")
plt.plot(df_2["total_samples"], df_2["reward"], label="line 2")
plt.legend()
plt.show()
