# Compare Episodic and Step-based Reinforcement Learning

Author: Gong, Yuhe


## For training environment

python train.py --algo ppo --env_id ALRBallInACupSimpleDense-v0

python train.py --algo ppo --env_id DeepMindBallInCupDense-v0

python train.py --algo ppo --env_id HoleReacherDense-v0

## For continue training

python train_continue.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 1

python train_continue.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 1

## For enjoy a well-trained model:

python enjoy.py --algo ppo --env_id ALRBallInACupSimpleDense-v0 --model_id 1 --step 1000

python enjoy.py --algo ppo --env_id DeepMindBallInCupDense-v0 --model_id 18 --step 50

python enjoy.py --algo ppo --env_id HoleReacherDense-v0 --model_id 1 --step 400



