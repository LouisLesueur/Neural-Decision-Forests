from train import prepare_db, prepare_model, prepare_optim, train
import torch
import numpy as np
import os

# dndf parameters
batch_size=500
feat_dropout=0.3
n_tree=3
tree_depth=10
n_class=10
tree_feature_rate=0.5
lr=0.001 # sgd: 10, adam: 0.001
gpuid=0
jointly_training=False
epochs=10
report_every=10

#putting them in a table for saving
params = np.array([batch_size, feat_dropout, n_tree, tree_depth, n_class, tree_feature_rate, lr, gpuid, jointly_training, epochs, report_every])

#id of the current test
if len(os.listdir('results')) >= 1:
    id = np.sort(np.array(os.listdir('results')).astype(int))[-1]+1
else:
    id = 0

#path for saving results
path = 'results/'+str(id)
os.mkdir(path)
# GPU
cuda = gpuid >= 0
if gpuid >= 0:
    torch.cuda.set_device(gpuid)
else:
    print("WARNING: RUN WITHOUT GPU")

db = prepare_db()
model = prepare_model(feat_dropout, n_tree, tree_depth, tree_feature_rate, n_class, jointly_training, cuda)
optim = prepare_optim(model, lr)
train_Loss, test_Loss, test_Acc = train(model, optim, db, epochs, jointly_training, n_class, batch_size, cuda, report_every)

#saving
np.savetxt(path + '/params.txt',params,fmt='%.2f')
np.savetxt(path + '/train_Loss.txt',train_Loss,fmt='%.2f')
np.savetxt(path + '/test_Loss.txt',test_Loss,fmt='%.2f')
np.savetxt(path + '/test_Acc.txt',test_Acc,fmt='%.2f')
