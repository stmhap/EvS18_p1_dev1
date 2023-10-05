import torch
import torch.nn as nn

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ContractMethod = 'MP'):
        super(ContractingBlock, self).__init__()

        self.ContractMethod = ContractMethod
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.StridedConv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        skip = x  # store the output for the skip connection
        if(self.ContractMethod == 'MP'):
            x = self.maxpool(x)
        else: ## ContractMethod == 'StrConv'
            x = self.StridedConv(x)

        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  ExpandMethod = 'Tr'):
        super(ExpandingBlock, self).__init__()
        
        self.ExpandMethod = ExpandMethod

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.TransposeConv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.UpSample =  nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #print(x.shape)

        if(self.ExpandMethod == 'Tr'):
            x = self.TransposeConv(x)
        else: # ExpandMethod = 'Up'
            x = self.UpSample(x)
            
        #print(x.shape)
        #print(skip.shape)
        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1) 
        # Out channels double here, so next conv channel has conv(outCh*2, outCh)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)        
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, ContractMethod = 'MP', ExpandMethod = 'Tr'):
        super(UNet, self).__init__()
        
        self.contract1 = ContractingBlock(in_channels, 64, ContractMethod)
        self.contract2 = ContractingBlock(64, 128, ContractMethod)
        self.contract3 = ContractingBlock(128, 256, ContractMethod)
        # self.contract4 = ContractingBlock(256, 512) # - Error in sir's code
        self.convBlock = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.expand1 = ExpandingBlock(512, 256, ExpandMethod)
        self.expand2 = ExpandingBlock(256, 128, ExpandMethod)
        self.expand3 = ExpandingBlock(128, 64, ExpandMethod)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        # x, _ = self.contract4(x) # - error in sir's code
        x = self.convBlock(x)
        
        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        x = self.final_conv(x) 
        return x