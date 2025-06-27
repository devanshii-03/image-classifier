import argparse
import torch
from torch import nn, optim
import os
from utils import load_data
from model import create_model, train_model, save_checkpoint

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    
    # Add arguments
    parser.add_argument('data_dir', type=str, help='Directory of the image data')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        choices=['vgg16', 'vgg13', 'densenet121'],
                        help='Model architecture (vgg16, vgg13, or densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if save directory exists, if not create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    dataloaders, image_datasets = load_data(args.data_dir)
    
    # Create model
    print(f"Creating model with architecture {args.arch} and {args.hidden_units} hidden units...")
    model, arch = create_model(args.arch, args.hidden_units)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train model
    print("Training model...")
    model = train_model(model, dataloaders, criterion, optimizer, 
                        epochs=args.epochs, device=device)
    
    # Save checkpoint
    print("Saving model checkpoint...")
    save_checkpoint(model, args.save_dir, arch, optimizer, args.epochs, 
                    image_datasets['train'].class_to_idx)
    
    print("Training complete!")

if __name__ == '__main__':
    main()
