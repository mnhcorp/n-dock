import unittest
import torch
import tempfile
import os
from n_dock.pre_training import pre_train

class TestPreTraining(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to simulate ImageNet data
        self.test_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.test_dir, 'train'))
        
        # Create dummy class directories and images
        for i in range(10):  # Create 10 classes
            class_dir = os.path.join(self.test_dir, 'train', f'n{i:08d}')
            os.makedirs(class_dir)
            for j in range(5):  # Create 5 images per class
                img_path = os.path.join(class_dir, f'img_{j}.jpg')
                with open(img_path, 'w') as f:
                    f.write('dummy image data')

    def tearDown(self):
        # Remove the temporary directory after the test
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_pre_training(self):
        config = {
            'data_path': self.test_dir,
            'epochs': 1,
            'batch_size': 2,
            'in_channels': 3,
            'base_filters': 16,
            'n_blocks': 2,
            'learning_rate': 0.001
        }
        
        model = pre_train(config)
        
        # Check if the model is an instance of SimpleCNN
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check if the model has learned parameters
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertNotEqual(torch.sum(param.data), 0)

if __name__ == '__main__':
    unittest.main()
