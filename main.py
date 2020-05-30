from argparse import Namespace
from train import prepare_db, prepare_model, prepare_optim, train
import torch
import numpy as np
import os
import yaml

with open("config.yaml") as file:
    params = yaml.safe_load(file)
opt = Namespace(**params)
opt.cuda = opt.gpuid >= 0

if not 'results' in os.listdir():
    os.mkdir('results')

# id of the current test
if len(os.listdir('results')) >= 1:
    run_id = np.sort(np.array(os.listdir('results')).astype(int))[-1]+1
else:
    run_id = 0

# path for saving results
path = f"results/{run_id}/"
os.mkdir(path)
# GPU
if opt.cuda:
    torch.cuda.set_device(opt.gpuid)
else:
    print("WARNING: RUN WITHOUT GPU")

db = prepare_db()
model = prepare_model(opt)
optim = prepare_optim(model, opt)
train_Loss, test_Loss, test_Acc = train(model, optim, db, opt)

#saving
with open(f"{path}params.yaml", 'w') as file:
    yaml.safe_dump(params, file)
np.savetxt(path + 'train_Loss.txt', train_Loss, fmt='%.2f')
np.savetxt(path + 'test_Loss.txt',  test_Loss,  fmt='%.2f')
np.savetxt(path + 'test_Acc.txt',   test_Acc,   fmt='%.2f')
