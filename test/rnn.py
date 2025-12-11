import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

num_classes=10
num_epochs=1 
input_size=28
sequence_length=28
num_layers=2 
hidden_size=256
batch_size=64
lr=0.001

device="mps" if torch.mps.is_available() else "cpu"
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size 
        self.num_layers=num_layers 
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*sequence_length,num_classes)

    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

        out,_=self.rnn(x,h0)
        out=out.reshape(out.shape[0],-1)
        out=self.fc(out)
        return out 

train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


model=RNN(input_size,hidden_size,num_layers,num_classes).to(device)
Losser=nn.CrossEntropyLoss()
optimizer=optim.AdamW(model.parameters(),lr=lr)


print("/////////////////TRAINING\\\\\\\\\\\\\\\\\\\\\\")
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        #print("Shape before",data.shape)
        #data=data.reshape(data.shape[0],-1)
        #print("Shape after",data.shape)
        data=data.to(device).squeeze(1)
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

        

