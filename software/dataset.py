import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MNIST16x16Dataset(Dataset):
    def __init__(self, root="./data", is_train=True, hidden_dim=32):
        super().__init__()
        self.is_train = is_train
        self.hidden_dim = hidden_dim
        
        # As per Evaluation Task: downsize to 16x16, grayscale
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.dataset = torchvision.datasets.MNIST(
            root=root, train=is_train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img shape: (1, 16, 16). Squeeze to (16, 16) where seq_len=16
        x = img.squeeze(0)
        
        # Pad the feature dimension from 16 to the requested hidden_dim
        if self.hidden_dim > 16:
            pad_right = self.hidden_dim - 16
            x = torch.nn.functional.pad(x, (0, pad_right))
        elif self.hidden_dim < 16:
            x = x[:, :self.hidden_dim]
            
        return x, label

if __name__ == "__main__":
    # Test the dataset
    ds = MNIST16x16Dataset(is_train=True, hidden_dim=64)
    x, y = ds[0]
    print(f"Sample shape: {x.shape}, Label: {y}") # Expected: [16, 64]