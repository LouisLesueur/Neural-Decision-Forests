# MALAP - Deep-Neural-Decision-Forests
An implementation of the Deep Neural Decision Forests(dNDF) in PyTorch, forked from [this repo](https://github.com/jingxil/Neural-Decision-Forests) and adaptated for an academic project. It works with MNIST dataset.
![](http://cnyah.com/2018/01/29/dNDF/arch.png)

# Requirements
- Python 3.x
- PyTorch >= 1.0.0
- numpy
- sklearn


# Usage
Change the hyperparameters in
>config.yaml

To train the network:
 ```
 python main.py
 ```

To plot loss and accuracy (after training)
 ```
python plot.py
 ```

# Authors
+ **Romain Bertrand**
+ **Louis Hémadou**
+ **Louis Lesueur**
+ **François Medina**
