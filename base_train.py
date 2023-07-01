import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import gradio as gr
import torch
import argparse
import whisper
import numpy as np
from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_convert


class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)


    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Extract bounding boxes and class labels from the target
        bounding_boxes = [obj["bbox"] for obj in target]
        class_labels = [obj["category_id"] for obj in target]

        # Convert bounding boxes from (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max) format
        bounding_boxes = box_convert(torch.tensor(bounding_boxes), in_fmt="xywh", out_fmt="xyxy")

        # Convert class labels from a list to a tensor
        class_labels = torch.tensor(class_labels)

        # Replace the original target with the processed bounding boxes and class labels
        target = {"bounding_boxes": bounding_boxes, "class_labels": class_labels}

        return img, target

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args

# Load the dataset
data_dir = 'path/to/coco/data'
train_annotation_file = os.path.join(data_dir, 'annotations/instances_train2017.json')
train_img_dir = os.path.join(data_dir, 'train2017')

# Define the data augmentation and transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and data loader
train_dataset = CustomCocoDataset(root=train_img_dir, annFile=train_annotation_file, transform=data_transform)
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Initialize the model
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)
model = BaseModel(opt, build_model(opt)).eval().cuda()

# Define the optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define the loss function
# Replace this with the appropriate loss function for your model
loss_function = torch.nn.CrossEntropyLoss()

# Number of training epochs
num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    model.train()
    for i, (images, targets) in enumerate(train_data_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = loss_function(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

    # Update the learning rate
    lr_scheduler.step()

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
