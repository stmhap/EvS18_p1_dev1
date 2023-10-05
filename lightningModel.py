import pytorch_lightning as pl
import torch
from torch import nn
from model import UNet
from loss import dice_loss, dice_loss_fn



class CustomUnet(pl.LightningModule):
    def __init__(self, loss = 'CE', ContractMethod = 'MP', ExpandMethod = 'Tr'):
        super().__init__()

        self.model = UNet(3, 3, ContractMethod = 'MP', ExpandMethod = 'Tr')
        self.loss = loss     

        self.train_losses = []
        self.save_hyperparameters()

        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

   
    def training_step(self, batch, batch_idx):
        input_imgs, target_masks = batch
        pred_mask_prob =  self.model(input_imgs) 
        target_masks = target_masks*255 #PIL Image open divides all values by 255

             
        # Reshape the predicted output tensor to [batch_size * height * width, num_classes]
        predicted_output = pred_mask_prob.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        # cross entropy works based on indices, so convert classes 1,2,3 to 0,1,2
        target_categories = target_masks - 1

        # Squeeze the singleton dimension from the target tensor and convert to long tensor
        target_labels = torch.squeeze(target_categories, dim=1).view(-1).long()

        if(self.loss == 'CE'): #Cross Entropy loss
            # Create the CrossEntropyLoss criterion
            criterion = torch.nn.CrossEntropyLoss()

            # Calculate the cross-entropy loss
            loss = criterion(predicted_output, target_labels)
            self.train_losses.append(loss.item())

            loss.backward(retain_graph=True)

            return loss
        
        else: #Dice loss

            # From the probability of predictions for 3 classes, construct the actual prediction of values - 1,2,3
            # max_channels = torch.argmax(pred_mask_prob, dim=1) + 1
            # result_tensor_predicted = max_channels.unsqueeze(1)

            #print(result_tensor_predicted.shape)
            loss = dice_loss_fn(predicted_output, target_labels, n_classes=3)
            
            self.train_losses.append(loss.item())
            
            loss.backward(retain_graph=True)

            return loss
                    
    
    def on_train_epoch_end(self):
        # print loss at the end of every epoch   
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        print(f'Mean training loss at end of epoch {self.trainer.current_epoch} = {mean_loss}')
        
