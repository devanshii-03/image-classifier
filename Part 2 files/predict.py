import argparse
import torch
import json
from PIL import Image
from utils import process_image, load_checkpoint, load_category_names
from model import predict

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    
    # Add arguments
    parser.add_argument('input', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device)
    
    # Load category names if provided
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
    else:
        cat_to_name = None
    
    # Make prediction
    print(f"Making prediction for image: {args.input}")
    probs, classes = predict(args.input, model, args.top_k, device)
    
    # Print results
    print("\nTop K predictions:")
    for i, (prob, class_idx) in enumerate(zip(probs, classes)):
        class_name = cat_to_name[class_idx] if cat_to_name else class_idx
        print(f"Rank {i+1}: {class_name} with probability {prob:.3f}")

if __name__ == '__main__':
    main()
