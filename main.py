from train import prepare_db, prepare_model, prepare_optim, train
import torch



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


# GPU
cuda = gpuid >= 0
if gpuid >= 0:
    torch.cuda.set_device(gpuid)
else:
    print("WARNING: RUN WITHOUT GPU")

db = prepare_db()
model = prepare_model(feat_dropout, n_tree, tree_depth, tree_feature_rate, n_class, jointly_training, cuda)
optim = prepare_optim(model, lr)
train(model, optim, db, epochs, jointly_training, n_class, batch_size, cuda, report_every)
