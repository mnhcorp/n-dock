import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from n_dock.models.simple_cnn import SimpleCNN
from n_dock.data_ingestion import data_ingest

def pre_train(pretrain_config):
    """
    Pre-train a foundation model using ingested data.
    
    Args:
    pretrain_config (dict): A dictionary containing configuration parameters.
    
    Returns:
    tuple: The pre-trained model and a function to generate embeddings.
    """
    # Data preparation
    data_config = {
        'data_type': pretrain_config['modality'],
        'data_path': pretrain_config.get('data_path', './data'),
        'image_size': pretrain_config.get('image_size', 224)
    }
    dataset = data_ingest(data_config)
    dataloader = DataLoader(dataset, batch_size=pretrain_config.get('batch_size', 32), shuffle=True, num_workers=4)
    
    # Model initialization
    if pretrain_config['architecture'] == 'SimpleCNN':
        model = SimpleCNN(
            in_channels=3,
            base_filters=pretrain_config.get('base_filters', 16),
            n_blocks=pretrain_config.get('n_blocks', 4)
        )
    elif pretrain_config['architecture'] == 'CLIP':
        raise NotImplementedError("CLIP architecture not yet implemented")
    elif pretrain_config['architecture'] == 'DINOv2':
        raise NotImplementedError("DINOv2 architecture not yet implemented")
    else:
        raise ValueError(f"Unsupported architecture: {pretrain_config['architecture']}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=pretrain_config.get('learning_rate', 0.001))
    
    # Training loop
    num_epochs = pretrain_config.get('epochs', 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            data, target = batch['image'].to(device), batch['label'].to(device)
            
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
    
    
    def get_embedding(input_data, model):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            embedding = model(input_data)  # Pass input data through the model
            # flatten to 1D
            embedding = embedding.view(embedding.size(0), -1)
            return embedding  # Return the output of the last layer
            
    return model, get_embedding
