import os

run = 20
for i in range(6):
    str = f'python cw2cmaes.py cw2cames/dmp_{i}.yml'
    os.system(str)
    str = f'python cw2cmaes.py cw2cames/promp_{i}.yml'
    os.system(str)
