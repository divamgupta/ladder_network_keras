
from keras.datasets import mnist
import keras
import random
from sklearn.metrics import accuracy_score

from ladder_net import get_ladder_network_fc

# get the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.0
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.0

y_train = keras.utils.to_categorical( y_train )
y_test = keras.utils.to_categorical( y_test )


# only select 100 training samples 
idxs_annot = range( x_train.shape[0])
random.seed(0)
random.shuffle( idxs_annot )
idxs_annot = idxs_annot[ :100 ]

x_train_unlabeled = x_train
x_train_labeled = x_train[ idxs_annot ]
y_train_labeled = y_train[ idxs_annot  ]


n_rep = x_train_unlabeled.shape[0] / x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)


# initialize the model 
inp_size = 28*28 # size of mnist dataset 
n_classes = 10
model = get_ladder_network_fc( layer_sizes = [ inp_size , 1000, 500, 250, 250, 250, n_classes ]  )


# train the model for 100 epochs
for _ in range(100):
    model.fit([ x_train_labeled_rep , x_train_unlabeled   ] , y_train_labeled_rep , epochs=1)
    y_test_pr = model.test_model.predict(x_test , batch_size=100 )
    print "test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1)  )





