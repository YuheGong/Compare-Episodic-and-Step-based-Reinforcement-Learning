# Compare Episodic and Step-based Reinforcement Learning

Author: Gong, Yuhe

#Overview

This repository builds a framework to easily compare episodic and step-based reinforcement learning with several environments (DeepMind Control Suite , ALR environments, OpenAI environment).

- For step-based reinforcement learning, we use PPO, SAC from stable-baselines3 framework.

- For episodic reinforcement learning, we use DMP, ProMP from ALR's framework

We use different reward function to compare the performance in each environment:

- Sparse reward function: in one episode, only one step has the reward according to the task, other steps' rewards is 0.
- Dense reward function: in one episode, every step has the reward according to the task.

FixedTargetReacherEnv Highly
configurable modification of the MuJoCo Reacher-v2 environment.



Training
In order to train an agent, you need a config file. Several are provided in ./configs.
For example, to solve the MuJoCo Reacher with CMA-ES, run:
python -m t5 -c configs/reacher_cma.yml
After training is done, all training output plus the configuration file are saved in
the experiments directory, under a timestamped name. You can change output path in the
config file.

Testing
After training, the learned policy is evaluated on a single episode and rendered
automatically. If you want to evaluate a trained policy separately, you can run:
python -m t5 -m experiments/cma-reacher20210228-105448

TODOs

Saving videos
Decouple reward function from environment, set in config file









-------------------------------------------------------------
## For training environment

#### Step-based algo

python train.py --algo ppo --env_id ALRBallInACupSimpleDense-v0

python train.py --algo ppo --env_id DeepMindBallInCupDense-v0

python train.py --algo ppo --env_id HoleReacherDense-v0

#### Episodic algo

python train.py --algo cmaes --env_id DeepMindBallInCupDMP-v0 --stop_cri True

python train.py --algo cmaes --env_id DeepMindBallInCupDenseDMP-v0

python train.py --algo cmaes --env_id DeepMindBallInCupProMP-v0

python train.py --algo cmaes --env_id DeepMindBallInCupDenseProMP-v0


## For continue training

python train_continue.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 1

python train_continue.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 1

## For enjoy a well-trained model:

python enjoy.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 18 --step 1000

python enjoy.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 2 --step 300

python enjoy.py --algo cmaes --env_id DeepMindBallInCupDenseProMP-v0 --model_id 4 --step 300

python enjoy.py --algo ppo --env_id HoleReacherDense-v0 --model_id 1 --step 400



