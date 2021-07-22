# Compare Episodic and Step-based Reinforcement Learning

Author: Gong, Yuhe

#Overview

This repository builds a framework to easily compare episodic and step-based reinforcement learning with several environments (DeepMind Control Suite , ALR environments, OpenAI environment).

- For step-based reinforcement learning, we use PPO, SAC from stable-baselines3 framework.

- For episodic reinforcement learning, we use DMP, ProMP from ALR's framework

We use different reward function to compare the performance in each environment:

- Sparse reward function: in one episode, only one step has the reward according to the task, other steps' rewards is 0.
- Dense reward function: in one episode, every step has the reward according to the task.

# Training with dense reward

|Name (env_id)| PPO|SAC|DMP|ProMP
|---|---|---|---|---|
|`ALR HoleReacher`|:heavy_check_mark:|  |  | 
|`ALR Ball In A Cup`|:heavy_check_mark:|  |  | 
|`DeepMind Ball In Cup`|:heavy_check_mark:|  | :heavy_check_mark:| :heavy_check_mark:

# Training with sparse reward

|Name (env_id)| PPO|SAC|DMP|ProMP
|---|---|---|---|---|
|`ALR HoleReacher`||  |  | 
|`ALR Ball In A Cup`||  |  | 
|`DeepMind Ball In Cup`||  | DeepMindBallInCupDMP-v0 | DeepMindBallInCupProMP-v0






