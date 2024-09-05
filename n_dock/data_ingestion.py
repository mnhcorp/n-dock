import os
from PIL import Image
import torch
from torchvision import transforms

def ingest_image_data(data_path, image_size=224):
    """
    Ingest image data from the specified directory.
    
    Args:
    data_path (str): Path to the directory containing image files.
    image_size (int): Size to which images will be resized (default: 224).
    
    Returns:
    torch.Tensor: A tensor containing the processed images.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for filename in os.listdir(data_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(data_path, filename)
            try:
                with Image.open(img_path) as img:
                    img_tensor = transform(img)
                    images.append(img_tensor)
            except Exception as e:
                print(f"Error processing image {filename}: {str(e)}")
    
    if not images:
        raise ValueError("No valid images found in the specified directory.")
    
    return torch.stack(images)

def data_ingest(data_config):
    """
    Main data ingestion function that handles different data types.
    
    Args:
    data_config (dict): A dictionary containing configuration parameters.
    
    Returns:
    A dataset object compatible with PyTorch DataLoader.
    """
    if data_config['data_type'] == 'image':
        data = ingest_image_data(data_config['data_path'], data_config.get('image_size', 224))
        return ImageDataset(data)
    elif data_config['data_type'] == 'text':
        # Placeholder for text data ingestion
        raise NotImplementedError("Text data ingestion not yet implemented")
    else:
        raise ValueError(f"Unsupported data type: {data_config['data_type']}")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'image': self.data[idx], 'label': 0}  # Placeholder label
