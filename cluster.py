import os

for i in range(5):
    str = f'python train.py --algo sac --env Meta-pick-place-v2 --seed {i}'
    os.system(str)
