import os

for i in range(1,5):
    #str = f'python train.py --algo sac --env_id ALRReacherBalanceIP-v3 --seed {i}'
    str = f'python train.py --algo promp --env_id dmcWalkerDenseProMP-v0'
    #str = f'python train.py --algo promp --env_id FetchReacherProMP-v1 --seed {i}'
    os.system(str)
    #str = f'python cw2cmaes/cmaes_cw2.py cw2cmaes/promp_dense_{i}.yml -o'
    #os.system(str)


for i in range(5):
    #str = f'python train.py --algo sac --env_id ALRReacherBalanceIP-v3 --seed {i}'
    str = f'python train.py --algo promp --env_id dmcCheetahDenseProMP-v0'
    #str = f'python train.py --algo promp --env_id FetchReacherProMP-v1 --seed {i}'
    os.system(str)