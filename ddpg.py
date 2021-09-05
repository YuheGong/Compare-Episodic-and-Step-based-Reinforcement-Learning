import os




run = 3
for env_id in range(3):
    env_name = f"DeepMindBallInCup-v{env_id}"
    for i in range(run):
        str = 'python train.py --algo ddpg --env_id ' + env_name + f' --seed {i}'
        os.system(str)
    env_name = f"DeepMindBallInCupDense-v{env_id}"
    for i in range(run):
        str = 'python train.py --algo ddpg --env_id ' + env_name + f' --seed {i}'
        os.system(str)

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