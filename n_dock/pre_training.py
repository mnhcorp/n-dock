import torch
import torch.nn as nn
import torch.optim as optim
from n_dock.models.simple_cnn import SimpleCNN
from n_dock.data_ingestion import data_ingest

def pre_train(config):
    """
    Pre-train a simple CNN foundation model.
    
    Args:
    config (dict): A dictionary containing configuration parameters.
    
    Returns:
    The pre-trained model.
    """
    # Data ingestion
    data = data_ingest(config['data_config'])
    
    # Model initialization
    model = SimpleCNN(
        in_channels=config.get('in_channels', 3),
        base_filters=config.get('base_filters', 16),
        n_blocks=config.get('n_blocks', 4)
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE for self-supervised learning
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Training loop
    num_epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            # Forward pass
            embeddings = model(batch)
            
            # Self-supervised learning: reconstruct the input
            loss = criterion(embeddings, batch.view(batch.size(0), -1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(data) // batch_size)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model
