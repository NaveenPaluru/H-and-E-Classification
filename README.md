# H-and-E-Classification

#### This repository hosts a binary classification problem. Specifically,Clinical Heart Failure Detection Using Whole-Slide Images of H&E tissue. The CNN model used for this project is [ENet](https://arxiv.org/abs/1606.02147).

### Data

Download the data from [here](https://idr.openmicroscopy.org/webclient/?show=project-402)

## Python Files

**trn.py** is the run file to train the ENet model. **tst.py** is the testing file for evaluation, **enet.py** has the ENet model definition, **config.py** is the configuration file and **myDataset.py** is data iterator.

## Matlab Files

**makemetadata.m** will read the data from annatations.csv and prepares the labels. **arrangedata.m**  will read the images from specified directory and accumulates the data for training, validation and testing.
