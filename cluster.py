import os

for i in range(5):
    str = f'python train.py --algo promp --env Meta-promp-dense-soccer-v2 --seed {i}'
    os.system(str)

    str = f'python train.py --algo promp --env Meta-promp-soccer-v2 --seed {i}'
    os.system(str)



for i in range(5):
    str = f'python train.py --algo promp --env Meta-promp-dense-coffee-push-v2 --seed {i}'
    os.system(str)

    str = f'python train.py --algo promp --env Meta-promp-coffee-push-v2 --seed {i}'
    os.system(str)