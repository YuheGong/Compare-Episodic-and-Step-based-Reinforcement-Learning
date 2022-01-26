# Compare Episodic and Step-based Reinforcement Learning

Author: Gong, Yuhe

# Overview

This repository builds a framework to easily compare episodic and step-based reinforcement learning with several environments (DeepMind Control Suite , ALR environments, OpenAI environment).

- For step-based reinforcement learning, we use PPO, SAC from stable-baselines3 framework.

- For episodic reinforcement learning, we use DMP, ProMP from ALR's framework

We use different reward function to compare the performance in each environment:

- Sparse reward function: in one episode, only one step has the reward according to the task, other steps' rewards is 0.
- Dense reward function: in one episode, every step has the reward according to the task.

# Training with dense reward

|Name| PPO|SAC|DMP|ProMP
|---|---|---|---|---|
|`ALRHoleReacher`|:heavy_check_mark:|  |  | 
|`ALRBallInACup`|:heavy_check_mark:|  |  | 
|`DeepMindBallInCup`|:heavy_check_mark:|  | :heavy_check_mark:| :heavy_check_mark:

# Training with sparse reward

|Name| PPO|SAC|DMP|ProMP
|---|---|---|---|---|
|`ALRHoleReacher`||  |  | 
|`ALRBallInACup`||  |  | 
|`DeepMindBallInCup`||  | :heavy_check_mark: | :heavy_check_mark:

# Command
## For training environment
- Step-based algo

python train.py --algo ppo --env_id ALRBallInACupSimpleDense-v0

python train.py --algo ppo --env_id DeepMindBallInCupDense-v0

python train.py --algo ppo --env_id DeepMindBallInCup-v0

python train.py --algo sac --env_id DeepMindBallInCupDense-v0 --seed 0

python train.py --algo ppo --env_id HoleReacherDense-v0

- Episodic algo

python train.py --algo dmp --env_id DeepMindBallInCupDMP-v0 --stop_cri True

python train.py --algo dmp --env_id DeepMindBallInCupDenseDMP-v0

python train.py --algo promp --env_id DeepMindBallInCupProMP-v0

python train.py --algo promp --env_id DeepMindBallInCupDenseProMP-v0

python train.py --algo dmp --env_id ALRReacherBalanceDMP-v0

## For continue training

python train_continue.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 1

python train_continue.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 1

python train_continue.py --algo ppo --env_id ALRReacher-v0 --model_id 5

## For enjoy a well-trained model:

python enjoy.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 20 --step 300

python enjoy.py --algo dmp --env_id DeepMindBallInCupDenseDMP-v0 --model_id 2 --step 300

python enjoy.py --algo promp --env_id DeepMindBallInCupDenseProMP-v0 --model_id 4 --step 300

python enjoy.py --algo ppo --env_id DeepMindBallInCup-v0 --model_id 3 --step 400

python enjoy.py --algo sac --env_id DeepMindBallInCup-v0 --model_id 3 --step 400

python enjoy.py --algo ppo --env_id ALRReacherBalance-v0 --model_id 10 --step 400






