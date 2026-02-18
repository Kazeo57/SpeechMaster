import torch 
import torch.nn as nn
import torch.nn.functional as F

"""The downsampler is a part of our final ASR that downsample the wav 
input for a low representation but with enought important features"""

class ConvBlock(nn.Module):
    def __init__(self,num_classes):
        #ENtrée: 2d(batch,channel,w,h) 1d:(batch,channel,length)
        super.__init__()
        #self.conv1=nn.Conv2d(in_channels=,out_channels=kernel_size=(3,3),stride=(1,1),padding='same')
        self.pool=nn.MaxPool2d(kernel_size=(1,1),stride=(1,1))
        self.fc=nn.Linear(num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=self.fc(x)
        return x
    


###YOUTUBE VERSION

class ResidualDonwSampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride,kernel_size=4):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels,
            out_channels,
            kernel_size,
            padding='same'
        )

        self.bn1=nn.BatchNorm1d(out_channels)
        
        self.conv2=nn.Conv1d(out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,

        )
    
    def forward(self,x):
        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu(output)+x 
        output=self.conv2(output)
        return output


class DownSampleNet(nn.Module):
    def __init__(self,in_channels,out_channels,strides,kernel_size,init_mean_pooling_kernel_size):
        super().__init__()
        self.layers=[ResidualDonwSampleBlock(in_channels,out_channels,strides,kernel_size) for _ in range(strides)]
        self.avg_pool=nn.AvgPool1d(kernel_size=init_mean_pooling_kernel_size)
        self.final_conv=nn.Conv1d(
            in_channels,out_channels,kernel_size,padding='same'
        )
    def forward(self,x):
        x=self.avg_pool(x)
        for layer in self.layers:
            x=layer(x)

        x=self.final_conv(x)
        x=x.transpose(1,2)
        return x
    





