import os
from os.path import join
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from torchDataUtils import create_dataset, ApplyTransform
from torchTrainUtils import train

print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
torch.manual_seed(42)


# Define project/data directory/file paths
models_dir = "models"
model_dir = "model_91.3751" 

project_dir = os.getcwd()
data_dir_1 = join(project_dir, "dog-breeds-data", "images", "Images")
csv_file_1 = join(project_dir, "dog-breeds-data", "index.csv")
# data_dir_2 = join(project_dir, "dog-breeds-data-2", "classes")
# csv_file_2 = join(project_dir, "dog-breeds-data-2", "index.csv")
hp_file = join(project_dir, models_dir, model_dir, "hyperparameters.json")

print("project_dir:", '\t', project_dir)
print("\ndata_dir_1:", '\t', data_dir_1)
print("csv_file_1:", '\t', csv_file_1)
# print("\ndata_dir_2:", '\t', data_dir_2)
# print("csv_file_2:", '\t', csv_file_2)


# load model hyperparameters
hp_dict = None

with open(join(project_dir, models_dir, model_dir, "hyperparameters.json"), 'r') as f:
    hp_dict = json.load(f)

# Set model hyperparameters
size = hp_dict["size"]
batch_size = hp_dict["batchSize"]
n_epochs = 5
learning_rate = 0.000001

# Create dataset from dataframe
full_dataset = create_dataset(project_dir=project_dir,
                                                   csv_files=[csv_file_1])

# Define 80-20 split for train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Perform train test split
print("\nSplit data into train and test sets")
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply any data augmentations to train and test sets
print("Apply data augmentations to train and test data")
train_dataset = ApplyTransform(dataset=train_dataset, size=size, transform="train")
test_dataset = ApplyTransform(dataset=test_dataset, size=size, transform="test")

print("Create train and test dataloaders")
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
test_dataloader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False)

classes = full_dataset.classes
num_classes = len(classes)

print(f"\n{num_classes} Classes:")
print(classes)

# Load base model
model = torchvision.models.inception_v3(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Remove fully connected layer
num_ftrs = model.fc.in_features
model.aux_logits = False
fc = nn.Sequential(
    nn.Linear(num_ftrs, num_classes)
)
model.fc = fc

mps_device = torch.device("mps")
model.to(mps_device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# restore states of model and optimizer
state = torch.load(join(project_dir, models_dir, model_dir, "model.pt"))
model.load_state_dict(state["model_state_dict"])
# optimizer.load_state_dict(state["optimizer_state_dict"])

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Train model
history = train(model=model,
                input_size=size,
                classes=classes,
                train_dataloader=train_dataloader, 
                test_dataloader=test_dataloader,
                n_epochs=n_epochs,
                batch_size=batch_size,
                criterion=criterion, 
                optimizer=optimizer, 
                learning_rate=learning_rate,
                device=mps_device, 
                project_dir=project_dir,
                models_dir=models_dir,
                n_epochs_trained=hp_dict["totalEpochs"])