import torch
from donwsampler import ResidualDonwSampleBlock, DownSampleNet
def ASR():
    pass
    #conv
    #attention
    #rvq
    #text_predictor()
    

def main(x,in_channels,out_channels,strides,kernel_size,init_mean_pooling_kernel_size):
    print("Entry Shape",x.shape)
    x=DownSampleNet(in_channels,out_channels,strides,kernel_size,init_mean_pooling_kernel_size)(x)
    print("Doownsample x",x.shape)

in_channels=6
out_channels=12
strides=4
kernel_size=3
init_mean_pooling_kernel_size=3

if __name__=="__main__":
    x=torch.randn(3,6)
    main(x,in_channels,out_channels,strides,kernel_size,init_mean_pooling_kernel_size)


torch.nn.AvgPool1d()




###YOUTUBE VERSION