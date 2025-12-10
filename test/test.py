import torch 
import numpy as np
print('Torch version',torch.__version__)
array=[[1,4,9],[0,8,6]]
#Initialize Tensor 
tensor=torch.tensor(array)
print("Basic tensor",tensor)
device="mps" if torch.mps.is_available() else "cpu"
gpu_tensor=torch.tensor(array,dtype=torch.float32, device='mps') #cuda si le gpu tourne sur CUDA #requires_grad=True
print("Gpu tensor",gpu_tensor)

print("tensor elements types",tensor.dtype)
print("tensor device",tensor.device)
print("tensor grade",tensor.requires_grad)



x=torch.empty(size=(3,3))
print("Empty tensor",x)
x=torch.zeros((3,3))
print("Zeros tensor",x)
x=torch.rand((3,3))
print("Rand tensor",x)
x=torch.ones((3,3))
print("Ones tensor",x)
x=torch.eye(5,5)
print("Eyes tensor",x)
x=torch.diag(torch.ones(2))
print("Diagonal",x)

torch.empty(size=(1,5)).normal_(mean=0,std=1)
print("Normal tensor",x)
torch.empty(size=(1,5)).uniform_(0,1)
print("Uniformized tensor",x)

array=[[1,4,9],[0,8,6]]
np_array=np.array(array)
print("Array",array)
array_to_tensor=torch.from_numpy(np_array) # Poure revenir en arrière tensor.numpy()
print("Tensor got",array_to_tensor)

# Addition 
y=torch.tensor([2,8,1])
z=torch.tensor([2,8,0])
ad=torch.add(y,z)
print("Addition",ad)
#Substraction 
sub=y-z
print("Substration tensor",sub)
#Division 
d=torch.true_divide(z,y)
print("Division tensor",d)
#inplace operations

#Add element 
print("Add second element",y.add_(z))

print("Power(Exposant)",z.pow(2))

torch.rand()

x.view()
x.permute()
x.unsqueeze()