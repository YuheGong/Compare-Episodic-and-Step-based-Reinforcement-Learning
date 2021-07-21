# Compare Episodic and Step-based Reinforcement Learning

Author: Gong, Yuhe


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

python enjoy.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 1 --step 1000

python enjoy.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 18 --step 50

python enjoy.py --algo cmaes --env_id DeepMindBallInCupDenseProMP-v0 --model_id 4 --step 300

python enjoy.py --algo ppo --env_id HoleReacherDense-v0 --model_id 1 --step 400



