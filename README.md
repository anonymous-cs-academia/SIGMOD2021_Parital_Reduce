# Partial Reduce


## Requirements

Install a latest [pytorch](https://pytorch.org) python environment.

## Data preprocessing

Download the cifar10 dataset and split it for the workers.

```
cd data
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz
cd ..
python dataset_split.py
```

## Training

For constant partial reduce with P=3:

```
./run_con_p_3.sh
```

For dynamic partial reduce with P=3:

```
./run_dyn_p_3.sh
```

Our prototype system is built on a real production cluster with our industrial partner. Due to the non-disclosure agreement, here we provide a multi-process based simplified demo to show the reproducibility of our approach. Each process can be regarded as a worker. When multiple processes are assigned to the same GPU, it can simulate real resouce sharing scenario and achieve similar performane.
