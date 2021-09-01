import os

'''
run = 20
for i in range(6):
    str = f'python cmaes_cw2.py cw2cames/dmp_{i}.yml'
    os.system(str)
    str = f'python cmaes_cw2.py cw2cames/promp_{i}.yml'
    os.system(str)
'''
run = 20
for j in range(run):
    for i in range(3):
        str = f'python train.py --algo cmaes --env_id DeepMindBallInCupDenseDMP-v{i} --seed {j} '

        os.system(str)
        str = f'python train.py --algo cmaes --env_id DeepMindBallInCupDMP-v{i} --seed {j} '
        os.system(str)