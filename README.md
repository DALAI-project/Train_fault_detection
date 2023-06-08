# Fault detection model training

This repository contains code that was used for training the models for sticky note and folded corner detection. 

Fault detection is formulated as an image classification task, where a neural network model is trained to distinguish 
whether an image contains a specific fault or not. The neural network model has been built using the Pytorch library, 
and the model training is done by fine-tuning an existing [Densenet neural network model](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html). 

The code is split into three files: 

- `train.py` contains the main part of the code used for model training
- `utils.py` contains utility functions used for example for saving the model and plotting the training and validation metrics
- `augment.py` contains code for creating augmentations (f.ex. by using rotation, blurring and padding) of the input images

## Running the code in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

#### Create and activate conda environment using the following commands:

`conda create -n fault_detection_env python=3.7`

`conda activate fault_detection_env`

#### Install dependencies listed in the *requirements.txt* file:

`pip install -r requirements.txt`

#### Run the training code 

When using the default values for all of the model parameters, the training can be initiated from the command line by typing

`python train.py`

The different model parameters are explained in more detail below.

## Model parameters

### Data

By default, the code expects the images containing faults (for instance sticky notes or folded corners) and the images without faults to be located in separate folders.
In addition, train and validation data for both types of images is also expected to be located in separate folders.

Data parameters:
- `tr_data_folder` defines the folder where the training data containing faults is located. Default folder path is `./data/faulty/train/`
- `val_data_folder` defines the folder where the validation data containing faults is located. Default folder path is `./data/faulty/val/`
- `tr_ok_folder` defines the folder where the training data that does not contain faults is located. Default folder path is `./data/ok/train/`
- `val_ok_folder` defines the folder where the validation data that does not contain faults is located. Default folder path is `./data/ok/val/`

The accepted input image file types are .jpg, .png and .tiff. Pdf files should be transformed into one of these images formats before used as an input to the model.

### Saving the model and training results

Parameters:

- `results_folder` defines the folder where the plots of the training an validation metrics (loss, accuracy, F1-score) and learning rates are saved. Default folder path is `./results`
 
- `save_model_path` default="./models/",
                    help='Path for saving model file.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size used for model training. ')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Base learning rate.')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of classes used in classification.')
parser.add_argument('--num_epochs', type=int, default=15,
                    help='Number of training epochs.')
parser.add_argument('--random_seed', type=int, default=8765,
                    help='Number used for initializing random number generation.')
parser.add_argument('--early_stop_threshold', type=int, default=2,
                    help='Threshold value of epochs after which training stops if validation accuracy does not improve.')
parser.add_argument('--save_model_format', type=str, default='onnx',
                    help='Defines the format for saving the model.')
parser.add_argument('--augment_choice', type=str, default=None,
                    help='Defines which image augmentation(s) are used. Defaults to randomly selected augmentations.')
parser.add_argument('--date', type=str, default=time.strftime("%d%m%Y"),
                    help='Current date.')


## Data

By default, the 
