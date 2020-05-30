import matplotlib.pyplot as plt
import numpy as np


id = '0'
path = 'results/'+str(id)


params = np.loadtxt(path + '/params.txt')
train_Loss = np.loadtxt(path + '/train_Loss.txt')
test_Loss = np.loadtxt(path + '/test_Loss.txt')
test_Acc = np.loadtxt(path + '/test_Acc.txt')

plt.figure()
plt.plot(train_Loss, label = 'train loss')
plt.plot(test_Loss, label = 'test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title(f"Loss for {int(params[2])} trees of depth: {int(params[3])} and a batch size of {int(params[0])}")
plt.legend()
plt.show()


plt.figure()
plt.plot(test_Acc)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title(f"Accuracy for {int(params[2])} trees of depth: {int(params[3])} and a batch size of {int(params[0])}")
plt.show()
