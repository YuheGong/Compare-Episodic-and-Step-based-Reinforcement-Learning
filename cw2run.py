import os

for i in range(20):
    str = f'python train.py --algo sac --env_id FetchReacher-v0 --seed {i}'
    os.system(str)
    #str = f'python cw2cmaes/cmaes_cw2.py cw2cmaes/promp_dense_{i}.yml -o'
    #os.system(str)
