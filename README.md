# Semi-Supervised Learning with Ladder Networks in <u>Keras</u>

This is an implementation of Ladder Network in Keras. Ladder network is a model for semi-supervised learning. Refer to the paper titled [_Semi-Supervised Learning with Ladder Networks_](http://arxiv.org/abs/1507.02672) by A Rasmus, H Valpola, M Honkala,M Berglund, and T Raiko

The model achives **98%** test accuracy on MNIST with just **100 labeled examples**. 

The code only works with Tensorflow backend.


## Requirements

- Python 3.6+
- Tensorflow (2.0)
- sklearn 


## How to use

Create a virtual environment

```bash
$ python3 -m venv --prompt ladder venv
$ source venv/bin/activate
(ladder) $ python -m pip install --upgrade pip
(ladder) $ python -m pip install wheel
(ladder) $ python -m pip install -r requirements.txt
```

Run the MNIST training script

```bash
(ladder) $ python mnist_example.py
```
