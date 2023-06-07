from __future__ import print_function
from __future__ import division
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import  models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import time
import random
import argparse
from tqdm import tqdm
from PIL import Image, ImageFile
from pathlib import Path

from augment import RandAug
import utils

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Much of the code is a modified version of the code available at
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

parser = argparse.ArgumentParser('arguments for training')

parser.add_argument('--tr_data_folder', type=str, default="./data/faulty/train/",
                    help='path to training data with faulty images')
parser.add_argument('--val_data_folder', type=str, default="./data/faulty/val/",
                    help='path to validation data with faulty images')
parser.add_argument('--tr_ok_folder', type=str, default="./data/ok/train/",
                    help='path to training data with ok images')
parser.add_argument('--val_ok_folder', type=str, default="./data/ok/val/",
                    help='path to validation data with ok images')
parser.add_argument('--results_folder', type=str, default="results/",
                    help='Folder for saving training results.')
parser.add_argument('--save_model_path', type=str, default="./models/",
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

args = parser.parse_args()

# PIL settings to avoid errors caused by truncated and large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Detects if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# List for saving the names of damaged images
damaged_images = []
 
def get_datapaths():
  """Function for loading train and validation data."""
    tr_files = list(Path(args.tr_data_folder).glob('*'))
    tr_ok_files = list(Path(args.tr_ok_folder).glob('*'))
    val_files = list(Path(args.val_data_folder).glob('*'))
    val_ok_files = list(Path(args.val_ok_folder).glob('*'))
    # Create labels for train and validation data
    tr_labels = np.concatenate((np.ones(len(tr_files)), np.zeros(len(tr_ok_files))))
    val_labels = np.concatenate((np.ones(len(val_files)), np.zeros(len(val_ok_files))))
    # Combine faulty and non-faulty images
    tr_files = tr_files + tr_ok_files
    val_files = val_files + val_ok_files

    print('Training data with faulty images: ', len(tr_files))
    print('Training data without faulty images: ', len(tr_ok_files))

    print('Validation data with faulty images: ', len(val_files))
    print('Validation data without faulty images: ', len(val_ok_files))

    data_dict = {'tr_data': tr_files, 'tr_labels': tr_labels, 
                'val_data': val_files, 'val_labels': val_labels}

    return data_dict

class ImageDataset(Dataset):
    """PyTorch Dataset class is used for generating the training and validation datasets."""
    def __init__(self, img_paths, img_labels, transform=None, target_transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path)
            label = self.img_labels[idx]
        except:
            # Image is considered damaged if reading the image fails
            damaged_images.append(img_path)
            return None
        if self.transform:
            image = self.transform(image.convert("RGB"))
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def initialize_model():
    """Function for initializing pretrained neural network model."""
    model_ft = models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, args.num_classes)
    input_size = 224

    return model_ft, input_size


def collate_fn(batch):
    """Helper function for creating data batches."""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def initialize_dataloaders(data_dict, input_size):
    """Function for initializing datasets and dataloaders."""
    #Transformations for train and validation images
    data_transforms = RandAug(input_size, args.augment_choice)
    
    # Train and validation datasets 
    train_dataset = ImageDataset(img_paths=data_dict['tr_data'], img_labels=data_dict['tr_labels'],  transform=data_transforms)
    validation_dataset = ImageDataset(img_paths=data_dict['val_data'], img_labels=data_dict['val_labels'], transform=None)
    # Train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloaders_dict = {'train': train_dataloader, 'val': validation_dataloader}

    return dataloaders_dict


def get_criterion(data_dict):
    """Function for generating class weights and for initializing the loss function."""
    y = np.asarray(data_dict['tr_labels'])
    # Class weights are used for compensating the unbalance 
    # in the number of training data from the two classes
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights=torch.tensor(class_weights, dtype=torch.float)
    print('Class weights: ', class_weights)
     # Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    return criterion


def get_optimizer(model):
    """Function for initializing the optimizer."""
    # Model parameters are split into two groups: parameters of the classifier
    # layer and other model parameters
    params_1 = [param for name, param in model.named_parameters()
                if name not in ["classifier.weight", "classifier.bias"]]
    params_2 = model.classifier.parameters()
    # 10 x larger learning rate is used when training the parameters 
    # of the classification layers
    params_to_update = [
            {'params': params_1, 'lr': args.lr},
            {'params': params_2, 'lr': args.lr * 10}
            ]
    # Stochastic gradient descent optimizer
    #optimizer = torch.optim.SGD(params_to_update, args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(params_to_update, args.lr)
    # Scheduler reduces learning rate when validation accuracy does not improve for an epoch
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

    return optimizer, scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler=None):
    """Function for model training and validation."""
    since = time.time()
    tr_acc_history = []
    val_acc_history = []
    val_loss_history = []
    tr_loss_history = []
    lr1_history = []
    lr2_history = []
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_epoch = 0
    early_stop = False

    for epoch in tqdm(range(args.num_epochs)):
        lr1_history.append(optimizer.param_groups[0]["lr"])
        lr2_history.append(optimizer.param_groups[1]["lr"])
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in batch
            for inputs, labels in dataloaders[phase]:
                if dataloaders[phase] is None:
                    continue
                else:
                    inputs = inputs.to(device)
                    labels = labels.long().to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history only in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        # mode we calculate the loss by summing the final output and the auxiliary output
                        # but in testing we only consider the final output.

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # update statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            # Calculate loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('Epoch {} - {} - Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, phase, epoch_loss, epoch_acc))

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                if epoch_acc > best_acc:
                    # Weights of the model with best accuracy are copied and saved
                    utils.save_model(model, 224, args.save_model_format, args.save_model_path, args.date)
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc
                    best_epoch = epoch
                elif epoch - best_epoch > args.early_stop_threshold:
                    # terminates the training loop if validation accuracy has not improved
                    print("Early stopped training at epoch %d" % epoch)
                    # Set early stopping condition
                    early_stop = True
                    break  
            elif phase == 'train':
                tr_acc_history.append(epoch_acc)
                tr_loss_history.append(epoch_loss)

        # Break outer loop if early stopping condition is activated
        if early_stop:
            break

        if scheduler:
            scheduler.step(val_acc_history[-1])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Returns model with the weights from the best epoch (based on validation accuracy)
    #model.load_state_dict(best_model_wts)
    hist_dict = {'tr_acc': tr_acc_history, 
                 'val_acc': val_acc_history, 
                 'val_loss': val_loss_history,
                 'tr_loss': tr_loss_history,
                 'lr1': lr1_history,
                 'lr2': lr2_history}

    return hist_dict

def main():
    # Set random seed(s)
    utils.set_seed(args.random_seed)
    # Load image paths and labels
    data_dict = get_datapaths()
    # Initialize the model 
    model, input_size = initialize_model()
    # Print the model architecture
    #print(model_ft)
    # Send the model to GPU (if available)
    model = model.to(device)
    print("Initializing Datasets and Dataloaders...")
    dataloaders_dict = initialize_dataloaders(data_dict, input_size)
    criterion = get_criterion(data_dict)
    optimizer, scheduler = get_optimizer(model)
    # Train and evaluate model
    hist_dict = train_model(model, dataloaders_dict, criterion, optimizer, scheduler)
    print('Damaged images: ', damaged_images)
    #utils.save_model(model, input_size, args.save_model_format, args.save_model_path, args.date)
    utils.plot_metrics(hist_dict, args.results_folder, args.date)

main()
