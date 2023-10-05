import torch
import torchvision
import pytorch_lightning as pl
    
from torchvision import transforms    
    
class OxfordIIITPetsCustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir_train='data\\OxfordPets\\train', data_dir_test='data\\OxfordPets\\test', batch_size=16):
        super().__init__()       
        self.data_dir_train = data_dir_train
        self.data_dir_test = data_dir_test
        self.batch_size = batch_size

    def prepare_data(self):
        # Create a torchvision transform to resize the image
        target_size = (400,600)
        transform_resize_tensor = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()  # This converts the PIL Image to a PyTorch tensor
        ])

        #pre_transform = torchvision.transforms.ToTensor()
        #target_transform = torchvision.transforms.ToTensor()
        # Oxford IIIT Pets Segmentation dataset loaded via torchvision.       
        self.pets_train = torchvision.datasets.OxfordIIITPet(root=self.data_dir_train, split="trainval", target_types="segmentation", 
                                                             download=True, transform=transform_resize_tensor, target_transform=transform_resize_tensor)
        self.pets_test = torchvision.datasets.OxfordIIITPet(root=self.data_dir_test, split="test", target_types="segmentation", 
                                                            download=True, transform=transform_resize_tensor, target_transform = transform_resize_tensor)

    def setup(self,stage):
        pass
                 

    def train_dataloader(self):
        return  torch.utils.data.DataLoader(
            self.pets_train,
            batch_size= self.batch_size,
            shuffle=True,
            )
    
    def test_dataloader(self):
         return torch.utils.data.DataLoader(
            self.pets_test,
            batch_size= self.batch_size,
            shuffle=False,
            )
    
    def val_dataloader(self):
         return torch.utils.data.DataLoader(
            self.pets_test,
            batch_size=self.batch_size,
            shuffle=True,
            )
