import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

def load_data(data_dir):
    """
    Load and transform image data for training, validation, and testing
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transformations for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }
    
    return dataloaders, image_datasets

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array
    """
    # Open the image
    img = Image.open(image_path)
    
    # Resize
    img = img.resize((256, 256))
    
    # Center crop
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array
    np_image = np.array(img) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def load_checkpoint(filepath, device='cpu'):
    """
    Loads a checkpoint and rebuilds the model
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier = checkpoint['classifier']
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(weights='IMAGENET1K_V1')
        model.classifier = checkpoint['classifier']
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = checkpoint['classifier']
    else:
        raise ValueError(f"Architecture {checkpoint['arch']} not supported")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_category_names(json_file):
    """
    Load category names from JSON file
    """
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
