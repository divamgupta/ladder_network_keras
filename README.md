# Semi-Supervised Learning with Ladder Networks in <u>Keras</u>

This is an implementation of Ladder Network in Keras. Ladder network is a model for semi-supervised learning. Refer to the paper titled [_Semi-Supervised Learning with Ladder Networks_](http://arxiv.org/abs/1507.02672) by A Rasmus, H Valpola, M Honkala,M Berglund, and T Raiko

This implementation was used in the official code of our paper  [Unsupervised Clustering using Pseudo-semi-supervised Learning
](https://openreview.net/pdf?id=rJlnxkSYPS). The code can be found [here](https://github.com/divamgupta/deep_clustering_kingdra) and the blog post can be found [here](https://divamgupta.com/unsupervised-learning/2020/10/31/pseudo-semi-supervised-learning-for-unsupervised-clustering.html)

The model achives **98%** test accuracy on MNIST with just **100 labeled examples**. 

The code only works with Tensorflow backend.



## Requirements

- Python 2.7+/3.6+
- Tensorflow (1.4.0)
- numpy
- keras (2.1.4) 

Note that other versions of tensorflow/keras should also work.

## How to use

Load the dataset

```python
from keras.datasets import mnist
import keras
import random

# get the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.0
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.0

y_train = keras.utils.to_categorical( y_train )
y_test = keras.utils.to_categorical( y_test )

# only select 100 training samples 
idxs_annot = range( x_train.shape[0])
random.seed(0)
#random.shuffle( idxs_annot ) #TypeError: 'range' object does not support item assignment
random.shuffle( list(idxs_annot) )
idxs_annot = idxs_annot[ :100 ]

x_train_unlabeled = x_train
x_train_labeled = x_train[ idxs_annot ]
y_train_labeled = y_train[ idxs_annot  ]

```



Repeat the labeled dataset to match the shapes

```python
n_rep = x_train_unlabeled.shape[0] / x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)
```



Initialize the model

```python
from ladder_net import get_ladder_network_fc
inp_size = 28*28 # size of mnist dataset 
n_classes = 10
model = get_ladder_network_fc( layer_sizes = [ inp_size , 1000, 500, 250, 250, 250, n_classes ]  )
```



Train the model

```python
model.fit([ x_train_labeled_rep , x_train_unlabeled   ] , y_train_labeled_rep , epochs=100)
```



Get the test accuracy 

```python
from sklearn.metrics import accuracy_score
y_test_pr = model.test_model.predict(x_test , batch_size=100 )

print "test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1)  )
```
