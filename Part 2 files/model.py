import torch
from torch import nn, optim
from torchvision import models

def create_model(arch='vgg16', hidden_units=512):
    """
    Create a pre-trained model with a custom classifier
    """
    # Load a pre-trained network
    if arch == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        input_size = 25088
    elif arch == 'vgg13':
        model = models.vgg13(weights='IMAGENET1K_V1')
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        input_size = 1024
    else:
        raise ValueError(f"Architecture {arch} not supported")
    
    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # Define new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Replace the model classifier
    if arch.startswith('vgg'):
        model.classifier = classifier
    elif arch == 'densenet121':
        model.classifier = classifier
    
    return model, arch

def train_model(model, dataloaders, criterion, optimizer, epochs=10, device='cpu'):
    """
    Train the model
    """
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0
                model.train()
    
    return model

def save_checkpoint(model, save_dir, arch, optimizer, epochs, class_to_idx):
    """
    Save the model checkpoint
    """
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Model checkpoint saved to {save_dir}/checkpoint.pth")

def predict(image_path, model, topk=5, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model
    """
    # Process the image
    from utils import process_image
    import torch
    
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
    
    # Move model and image to the specified device
    model.to(device)
    img = img.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Calculate probabilities
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)
        top_p, top_indices = ps.topk(topk, dim=1)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_p[0].tolist(), top_classes
