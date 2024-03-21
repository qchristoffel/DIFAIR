# DIFAIR: Towards learning DIFferenciAted Image Representations

We proposed **DIFAIR**, a method aiming to control the representation space learned by a neural network. It associates each dimension of the representation space with a specific class. Moreover, extracted features should be distinct and activated only when the characteristic associated really is in the image, otherwise it should be ``disabled'', approaching a value close to 0. 

Having such features, we believe that it would enable the network to do Open-set recognition (OSR) [(Scheirer et al., 2013)](https://ieeexplore.ieee.org/abstract/document/6365193). This is a more realistic scenario of neural network usage, where classes that were never seen during training, *unknown classes*, can be presented to the model at test time. It is opposed to closed-set recognition where the same classes are present in the training and testing sets. The aim of OSR is to detect such *unknown classes* at test time, while being able to classify *known classes*.

To reach those objectives, we proposed to define class anchors in the representation space, around which we allocate a hypersphere for instances to be represented within. To enhance the distinction of extracted attributes, we added a loss term to penalize the correlation of filters of the last layer.

## Contents
[1. Setup](#setup)

[2. How to run experiments](#run)

[3. How to analyse results](#analyse)

[4. Experiments parameters](#parameters)


## <a name="setup"></a> 1. Setup

All the experiments where done in TensorFlow 2.12.1

### Environment setup
We wrote a script to setup the installation of all dependencies. `conda` is required for the creation of a virtual environment and the configuration of paths for GPU compatibility. 
``` bash
# Create an environment and install all required packages
cd setup/
bash create_env.sh
```

### Tiny ImageNet setup

In order to train models on Tiny ImageNet, it first need to be downloaded and transformed to be used as a [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

``` bash
cd datasets/tiny_imagenet
tfds build
```

## <a name="run"></a> 2. How to run experiments

### Running

The main file is `train_model.py`. \
Using flags, all experiments can be run.
All flags are described in `config_flags.py`, default flag values can be modified in this file. 

**Note:** depending on the loss chosen, different flags are used. 
For example, if `--loss=dist`, the values of `--max_dist` and `--anchor_multiplier` will be used, but not for the cross-entropy loss.

**Note:** if you want to run on specific class splits, they can be defined in `datasets/splits/osr_splits.py` then selected using the flag `--config <split_index>`.

---
### Default configs

Default configuration files, containing default values for runs on specific datasets are available in the directory `default_flags`. 
In those files, the loss is `difair` by default.

For example, if you want to run an experiment on CIFAR10 do:
``` bash
# By default it will run on the first split
python3 train_model.py --flagsfile defaults_flags/difair_cifar10_flags.txt
```

---

### Scripts

To run multiple experiments, we wrote scripts, available in the directory `python_scripts`.

For example, if you want to train models on all OSR datasets, on all splits do:
```bash
python3 python_scripts/benchmark_osr.py
```

Other scripts are documented at the beginning of each file.
By default, experiments are executed using bash, but it is possible to use slurm by setting SLURM=True in the files.

## <a name="analyse"></a> 3. How to analyse results

Multiple results analysis are possible:

**Benchmark results:**

The command below outputs accuracy, AUROC scores for the task of OSR, and compute the mean across splits for results located in `RES_DIR`, a variable to edit in the file.

``` bash
python3 benchmark_results.py
```

**Representations:**

The command below plots different types of graphics given a `flagfile`. This flagfile should describe the location of a model that will be loaded (given the flag `--save_path` and `--prefix`). \
Some additional flags are defined in this python file.

Generated graphics include: confusion matrix, tsne visualizations, mean representations of features, weights visualizations, class similarity matrixes.

``` bash
python3 analyse_results_model.py --flagfile <path_to_flag_file> --analyse output --plot_anchors --actualize_centers --save_format pdf
```

The notebook ```analyse_results_instance.ipynb``` can be used to generate visualization of representations for specific instances. 

**Hyperparameter & loss configuration search**

The notebooks ```hyperparameter_search.ipynb``` and ```loss_configurations_search.ipynb``` can be used to generate graphics in order to determine the best configurations for the models, given sets of results in `RES_DIR`.

## <a name="parameters"></a> 4. Experiments parameters

- RandAugment was used for all experiments, hyperparameters can be seen in default flags files.
- The learning rate scheduler used a cosine decay, with a warmup period of 10 epochs and learning rate restarts at 200 and 400 epochs.
- The maximum distance (i.e. the radius of the hypersphere) was set to 0.6 times the distance between two anchors.

**DIFAIR optimal hyperparameters**

| Dataset | Learning rate | Number of features | Anchor multiplier | Max distance | Batch size |
|---------|---------------|--------------------|-------------------|--------------|-----------|
| MNIST  | 0.3          | 5                  | 10.0              | 18.97        | 128        |
| SVHN   | 0.3          | 5                  | 10.0              | 18.97        | 128        |
| CIFAR10| 0.3          | 5                  | 10.0              | 18.97        | 128        |
| CIFAR+10 | 0.3         | 5                  | 10.0              | 18.97        | 128        |
| CIFAR+50| 0.3           | 5                  | 10.0              | 18.97        | 128        |
| Tiny ImageNet | 0.5   | 5                  | 10.0              | 18.97        | 128        |


