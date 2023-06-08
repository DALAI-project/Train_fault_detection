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

By default, the code expects the following folder structure:

```
├──fault_detection 
      ├──models
      ├──results 
      ├──data
      |   ├──faulty
      |   |   ├──train
      |   |   └──val
      |   └──ok
      |       ├──train
      |       └──val
      ├──train.py
      ├──utils.py
      ├──augment.py
      └──requirements.txt
```

Therefore the images containing faults (for instance sticky notes or folded corners) and the images without faults to be located in separate folders.
In addition, train and validation data for both types of images is also expected to be located in separate folders.

Parameters:
- `tr_data_folder` defines the folder where the training data containing faults is located. Default folder path is `./data/faulty/train`.
- `val_data_folder` defines the folder where the validation data containing faults is located. Default folder path is `./data/faulty/val`.
- `tr_ok_folder` defines the folder where the training data that does not contain faults is located. Default folder path is `./data/ok/train`.
- `val_ok_folder` defines the folder where the validation data that does not contain faults is located. Default folder path is `./data/ok/val`.

The parameter values can be set in command line when initiating training:

`python --tr_data_folder ./data/faulty/train --val_data_folder ./data/faulty/val --tr_ok_folder ./data/ok/train --val_ok_folder ./data/ok/val train.py`

The accepted input image file types are .jpg, .png and .tiff. Pdf files should be transformed into one of these images formats before used as an input to the model.

### Parameters related to saving the model and the training and validation results

The training performance is measured using training and validation loss, accuracy and F1 score (more information on the F1 score can be found for example [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)). The average of these values is saved each epoch, and the resulting values are plotted and saved in the folder defined by the user.

The trained model is saved by default after each epoch when the validation F1 score improves the previous top score. The model can be saved either in the [ONNX](https://onnx.ai/) format that is not dependent on specific frameworks like PyTorch and is optimized for inference speed, or by using PyTorch's default format for saving the model in serialized form. In the first instance, the model is saved as `densenet_date.onnx` and in the latter instance as `densenet_date.pth`. Date refers to the current date, so that a model trained on 7.6.2023 would be saved in the ONNX format as `densenet_07062023.onnx`.

Parameters:
- `results_folder` defines the folder where the plots of the training an validation metrics (loss, accuracy, F1-score) and learning rates are saved. Default folder path is `./results`.
- `save_model_path` defines the folder where the model file is saved. Default folder path is `./models`.
- `save_model_format` defines the format in which the model is saved. The available options are PyTorch (`torch`) and ONNX (`onnx`) formats. Default format is `onnx`.

The parameter values can be set in command line when initiating training:

`python --results_folder ./results --save_model_path ./models/ --save_model_format onnx train.py`

### Parameters related to model training

A Number of parameters are used for defining the conditions for model training. 

Learning rate defines how much the model weights are tuned after each iteration based on the gradient of the loss function. In the code, there are different learning rates for the classification layer and the pretrained layers of the base model. The `lr` parameter defines the learning rate for the base model layers, and the learning rate for the classification layer is automatically set to be 10 times larger.

Batch size defines the number of images that are processed before the model weights are updated. Number of epochs, on the other hand, defines how many times during the training the model goes through the entire training dataset. Early stopping is a method used for reducing overfitting by stopping training after a specific learning metric (loss, accuracy etc.) has not improved during a defined number of epochs.

Random seed parameter is used for setting the seed for initializing random number generation. This makes the training results reproducible when using the same seed, model and data. 

The `device` parameters defines whether cpu or gpu is used for model training.

Parameters:
- `lr` defines the learning rate used for adjusting the weights of the base model layers. The learning rate for the classification layer is always 10 times larger. Default value for the base learning rate is `0.0001`.
- `batch_size` defines the number of images in one batch. Default batch size is `16`.
- `num_epochs` sets the number of times the model goes through the entire training dataset. Default value is `15`.
- `early_stop_threshold` defines the number of epochs that training can go on without improvement in the chosen metric (validation F1 score by default). Default value is `2`.
-  `random_seed` sets the seed for initializing random number generation. Default value is `8765`.
-  `device` defines whether cpu or gpu is used for model training. Value can be for example `cpu`, `cuda:0` or `cuda:1`, depending on the specific gpu that is used.

The parameter values can be set in command line when initiating training:

`python --lr 0.0001 --batch_size 16 --num_epochs 15 --early_stop_threshold 2 --random_seed 8765 --device cpu train.py`

### Parameter for data augmentation

Data augmentations are used for increasing the diversity of the data and thus for helping to reduce overfitting. The available augmentation options are
- `identity`: This augmentation option only resizes the image to the required model input size (224 x 224) and transforms it into a PyTorch tensor form. This is the choice when no augmentations should be applied during model training.
- `rotate`: Image is rotated randomly between zero and 180 degrees. 
- `color`: The brightness, hue, contrast and saturation values of the image are transformed randomly on a defined scale. 
- `sharpness`: The sharpness of the image is transformed randomly on a defined scale.
- `blur`: The blurriness of the image is transformed randomly on a defined scale.
- `pad`: Padding of 3, 10 or 25 pixels is added to all sides of the image. The color of the padding is either black or white.
- `perspective`: Transforms the perspective of the image based on randomly chosen values from a defined scale.
- `None`: This option selects randomly an augmentation for each image from the above list. The options are weighted so that 'identity' is chosen with 40% probability, while each of the other augmentations has 10% probability of being selected.

More information and examples of the different image transform options are available [here](https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py).

Parameter:
-  `augment_choice` defines which image augmentation(s) are used during model training. Default value is `None`.  

The parameter value can be set in command line when initiating training:

`python --augment_choice None train.py`
