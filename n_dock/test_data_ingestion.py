import unittest
import os
import tempfile
from PIL import Image
import torch
from n_dock.data_ingestion import ingest_image_data, data_ingest

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create some test images
        self.image_size = 100
        self.num_images = 5
        for i in range(self.num_images):
            img = Image.new('RGB', (self.image_size, self.image_size), color = (73, 109, 137))
            img.save(os.path.join(self.test_dir, f'test_image_{i}.jpg'))

    def tearDown(self):
        # Remove the directory after the test
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_ingest_image_data(self):
        result = ingest_image_data(self.test_dir, image_size=224)
        
        # Check if the result is a torch.Tensor
        self.assertIsInstance(result, torch.Tensor)
        
        # Check if the shape is correct
        self.assertEqual(result.shape, (self.num_images, 3, 224, 224))

    def test_data_ingest(self):
        config = {
            'data_type': 'image',
            'data_path': self.test_dir,
            'image_size': 224
        }
        result = data_ingest(config)
        
        # Check if the result is a Dataset
        self.assertIsInstance(result, torch.utils.data.Dataset)
        
        # Check if the dataset has the correct length
        self.assertEqual(len(result), self.num_images)
        
        # Check if the first item has the correct format
        first_item = result[0]
        self.assertIn('image', first_item)
        self.assertIn('label', first_item)
        self.assertIsInstance(first_item['image'], torch.Tensor)
        self.assertEqual(first_item['image'].shape, (3, 224, 224))

    def test_invalid_data_type(self):
        config = {
            'data_type': 'invalid',
            'data_path': self.test_dir
        }
        with self.assertRaises(ValueError):
            data_ingest(config)

if __name__ == '__main__':
    unittest.main()
