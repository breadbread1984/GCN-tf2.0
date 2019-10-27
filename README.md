# GCN-tf2.0
this project implements graph convolutional networks with tensorflow 2.0. graph convolutional networks is a semi-supervised network which explicit represent the constraints with edges between nodes(samples). during training the only partial nodes(samples) need to be given. the network will evolve to guess the right label for the unknown samples.

## How to train on CORA
launch the training with

```python
python3 train_cora.py
```


