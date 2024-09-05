import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from n_dock.models.simple_cnn import SimpleCNN

def pre_train(config):
    """
    Pre-train a simple CNN foundation model using ImageNet data.
    
    Args:
    config (dict): A dictionary containing configuration parameters.
    
    Returns:
    The pre-trained model.
    """
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageNet(root=config['data_path'], split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=4)
    
    # Model initialization
    model = SimpleCNN(
        in_channels=config.get('in_channels', 3),
        base_filters=config.get('base_filters', 16),
        n_blocks=config.get('n_blocks', 4)
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model
