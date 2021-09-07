import os


run = 1
for i in range(3):
    str = f'python cmaes_cw2.py cw2cmaes/dmp_dense_{i}.yml -o'
    os.system(str)
    str = f'python cmaes_cw2.py cw2cmaes/promp_dense_{i}.yml'
    os.system(str)

'''
run = 1
for j in range(run):
    for i in range(3):
        str = f'python train.py --algo cmaes --env_id DeepMindBallInCupDenseDMP-v{i} --seed {j} '

        os.system(str)
        str = f'python train.py --algo cmaes --env_id DeepMindBallInCupDMP-v{i} --seed {j} '
        os.system(str)
'''