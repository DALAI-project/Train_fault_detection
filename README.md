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

### Parameters related to training and validation data

By default, the code expects the images containing faults (for instance sticky notes or folded corners) and the images without faults to be located in separate folders.
In addition, train and validation data for both types of images is also expected to be located in separate folders.

Parameters:
- `tr_data_folder` defines the folder where the training data containing faults is located. Default folder path is `./data/faulty/train/`.
- `val_data_folder` defines the folder where the validation data containing faults is located. Default folder path is `./data/faulty/val/`.
- `tr_ok_folder` defines the folder where the training data that does not contain faults is located. Default folder path is `./data/ok/train/`.
- `val_ok_folder` defines the folder where the validation data that does not contain faults is located. Default folder path is `./data/ok/val/`.

The parameter values can be set in command line when initiating training:

`python --tr_data_folder ./data/faulty/train/ --val_data_folder ./data/faulty/val/ --tr_ok_folder ./data/ok/train/ --val_ok_folder ./data/ok/val/ train.py`

The accepted input image file types are .jpg, .png and .tiff. Pdf files should be transformed into one of these images formats before used as an input to the model.

### Parameters related to saving the model and the training and validation results

The training performance is measured using training and validation loss, accuracy and F1 score (more information on the F1 score can be found for example [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)). The average of these values is saved each epoch, and the resulting values are plotted and saved in the folder defined by the user.

The trained model is saved by default after each epoch when the validation F1 score improves the previous top score. The model can be saved either in the [ONNX](https://onnx.ai/) format that is not dependent on specific frameworks like PyTorch and is optimized for inference speed, or by using PyTorch's default format for saving the model in serialized form. In the first instance, the model is saved as `densenet_date.onnx` and in the latter instance as `densenet_date.pth`. Date refers to the current date, so that a model trained on 7.6.2023 would be saved in the ONNX format as `densenet_07062023.onnx`.

Parameters:
- `results_folder` defines the folder where the plots of the training an validation metrics (loss, accuracy, F1-score) and learning rates are saved. Default folder path is `./results`.
- `save_model_path` defines the folder where the model file is saved. Default folder path is `./models/`.
- `save_model_format` defines the format in which the model is saved. The available options are PyTorch (`torch`) and ONNX (`onnx`) formats. Default format is `onnx`.

### Parameters related to model training

A Number of parameters are used for defining the conditions for model training. 

Learning rate defines how much the model weights are tuned after each iteration based on the gradient of the loss function. In the code, there are different learning rates for the classification layer and the pretrained layers of the base model. The `lr` parameter defines the learning rate for the base model layers, and the learning rate for the classification layer is automatically set to be 10 times larger.

Batch size 

Parameters:
- `lr` defines the learning rate used for adjusting the weights of the base model layers. The learning rate for the classification layer is always 10 times larger. Default value for the base learning rate is `0.0001`.
- 

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

parser.add_argument('--augment_choice', type=str, default=None,
                    help='Defines which image augmentation(s) are used. Defaults to randomly selected augmentations.')



## Data

By default, the 
