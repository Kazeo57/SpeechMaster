import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

num_classes=10
num_epochs=1
input_size=784
batch_size=64
lr=0.001

device="mps" if torch.mps.is_available() else "cpu"
class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.c1=nn.Linear(16*7*7,num_classes)
        

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)

        x=F.relu(self.conv2(x))
        x=self.pool(x)
        
        x=x.reshape(x.shape[0],-1)
        x=self.c1(x)
        #print("OUTTTTTTT",x)
        return x


train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


model=CNN().to(device)
Losser=nn.CrossEntropyLoss()
optimizer=optim.AdamW(model.parameters(),lr=lr)


print("/////////////////TRAINING\\\\\\\\\\\\\\\\\\\\\\")
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        print("Shape before",data.shape)
        #data=data.reshape(data.shape[0],-1)
        print("Shape after",data.shape)
        data=data.to(device)
        targets=targets.to(device)
        #print("vvvvvvvvv")
        results=model(data)
        #print("ttttttttttt")
        #forward  
        loss=Losser(results,targets)
        optimizer.zero_grad()
        #print("ppppppppp")
        #backward
        loss.backward()
        optimizer.step()
        #print("ooooooooo")

        

