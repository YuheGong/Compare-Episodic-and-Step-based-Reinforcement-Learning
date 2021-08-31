import os


run = 20
for i in range(run):
    os.system(f'python train.py --algo ppo --env_id DeepMindBallInCup-v2 --seed {i}')

"""
str=('python train.py --algo ppo --env_id DeepMindBallInCupDense-v0')

run = 3
for i in range(run):
    os.system(str)


str=('python train.py --algo ppo --env_id DeepMindBallInCup-v0')

run = 3
for i in range(run):
    os.system(str)


str=('python train.py --algo ppo --env_id DeepMindBallInCupDense-v2')

run = 3
for i in range(run):
    os.system(str)


str=('python train.py --algo ppo --env_id DeepMindBallInCup-v2')

run = 20
for i in range(run):
    os.system('python train.py --algo ppo --env_id DeepMindBallInCup-v2')


str=('python train.py --algo ppo --env_id DeepMindBallInCupDense-v1')

run = 1
for i in range(run):
    os.system(str)


str=('python train.py --algo ppo --env_id DeepMindBallInCup-v1')

run = 1
for i in range(run):
    os.system(str)
"""