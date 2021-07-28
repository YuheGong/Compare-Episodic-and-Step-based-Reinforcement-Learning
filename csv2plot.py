import pandas as pd
import plotly.express as px

df = pd.read_csv('/home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning/rosenbrock/log/rep_00/rep_0.csv')


print(df["total_samples"])
fig = px.line(df, x = "total_samples", y = "reward", title='Apple Share Prices over time (2014)')
fig.show()