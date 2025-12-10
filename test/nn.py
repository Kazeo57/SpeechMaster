import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

#Create MOdel
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.c1=nn.Linear(input_size,50)
        self.c2=nn.Linear(50,num_classes)
    #forward à la place de call dans tensorflow
    def forward(self,x):
        x=F.relu(self.c1(x))
        x=self.c2(x)
        return x
    

device="mps" if torch.mps.is_available() else "cpu"
input_size=784 
num_classes=10 
lr=0.001 
batch_size=64 
num_epochs=1 


#Load data

print(">>>>>>>>>>>>>>>>>>CHARGER LA DONNEE<<<<<<<<<<<<<<<<<<<<<<")
train_dataset=datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

model=NN(input_size=input_size,num_classes=num_classes).to(device)

loss_core=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=lr)

print(">>>>>>>>>>>>>>>>>>TRAINING<<<<<<<<<<<<<<<<<<<<<<")
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data=data.to(device=device)
        targets=targets.to(device=device)
        print("Data shape before reshaping",data.shape)
        data=data.reshape(data.shape[0],-1)
        print("Data Shape after reshaping",data.shape)
        #forward
        scores=model(data)

        #Compute loss and Initialize optimizer
        loss=loss_core(scores,targets)
        optimizer.zero_grad()
        #Apply Loss and optimizer on backpropagation
        loss.backward()
        optimizer.step()

def check_accuracy(loader,model):
    
    num_correct=0
    num_samples=0 
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)

            
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f"Eval End with accuracy of {num_correct/num_samples}")

    model.train()
 


check_accuracy(train_loader,model)
check_accuracy  (test_loader,model)

#Initialize network 

#Loss and optimizer 

#Train Network

#Evaluate