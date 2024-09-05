import torch
import torch.nn as nn
import torch.optim as optim
from n_dock.models.simple_cnn import SimpleCNN
from n_dock.data_ingestion import data_ingest

def pre_train(config):
    """
    Pre-train a simple CNN model for image classification.
    
    Args:
    config (dict): A dictionary containing configuration parameters.
    
    Returns:
    The pre-trained model.
    """
    # Data ingestion
    data = data_ingest(config['data_config'])
    
    # Model initialization
    model = SimpleCNN(num_classes=config.get('num_classes', 10))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    
    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, torch.zeros(batch.size(0)).long())  # Dummy labels
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model
