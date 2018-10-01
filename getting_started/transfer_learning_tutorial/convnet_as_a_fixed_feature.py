# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Transform
data_transforms = {
  'train' : transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
  'val' : transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
}

data_dir = "hymenoptera_data"
image_datasets = {
  x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
  for x in ['train', 'val']
}
dataloaders = {
  x : torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
  for x in ['train', 'val']
}
dataset_sizes = {
  x : len(image_datasets[x])
  for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Visualize a few images
def imshow(inp, title=None):
  """ Inshow for Tensors """
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.savefig("results/01.png")


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


# Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print("Epoch {} / {}".format(epoch, num_epochs - 1))
    print("-" * 10)

    # Each Epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == "train":
        scheduler.step()
        model.train() # Set model to training mode
      else:
        model.eval()
      
      running_loss = 0.0
      running_corrects = 0

      # Iterate over data
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        # Track the history if only in train
        with torch.set_grad_enabled(phase == "train"):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          """

          torch.max(outputs, 1)が返すもの
          
          outputs :
          tensor([[ 0.4015,  0.4264],
          [ 0.5723, -0.2256],
          [ 0.3828, -0.4551],
          [ 0.6952, -0.6357]], grad_fn=<ThAddmmBackward>)

          torch.max(outputs, 1)
          (tensor([0.4264, 0.5723, 0.3828, 0.6952], grad_fn=<MaxBackward0>), tensor([1, 0, 0, 0]))
          1つ目のTensorはnn.Linear(in_features, 2)での最後の層の結果を,
          二つ目のTensorはどちらのindexが大きいかを取得している

          """
          loss = criterion(outputs, labels) # calculate the loss

          # backward + optimize
          if phase == "train":
            loss.backward()
            optimizer.step()
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)
      
      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print("{} Loss : {:.4f} Acc : {:.4f}".format(
        phase, epoch_loss, epoch_acc
      ))
      
      # deep copy the model
      if phase == "val" and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()
  
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model


# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# ConvNet as a fixed feature extractor
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
  param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_fits = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_fits, 2)

model_conv.to(device)
crietion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# train model
model_conv = train_model(model_conv, crietion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)