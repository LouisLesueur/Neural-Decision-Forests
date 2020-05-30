from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import yaml


id = '0'
path = 'results/'+str(id)


with open(path + '/params.yaml') as file:
    opt = Namespace(**yaml.safe_load(file))
train_Loss = np.loadtxt(path + '/train_Loss.txt')
test_Loss = np.loadtxt(path + '/test_Loss.txt')
test_Acc = np.loadtxt(path + '/test_Acc.txt')

message = f"{opt.n_trees} trees of depth: {opt.tree_depth} and a batch size of {opt.batch_size}"

plt.figure()
plt.plot(train_Loss, label = 'train loss')
plt.plot(test_Loss, label = 'test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title(f"Loss for {message}")
plt.legend()
plt.show()


plt.figure()
plt.plot(test_Acc)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title(f"Accuracy for {message}")
plt.show()
