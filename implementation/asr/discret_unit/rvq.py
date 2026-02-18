import torch
import torch.nn as nn 
from sklearn.cluster import KMeans 

class RVQ(nn.Module):
    def __init__(self):
        super(RVQ,self).__init__()
        self.codebook={}
        self.kmeans=KMeans(n_clusters=4)
        
        

    def forwrad(self,frames_numpy):
        self.kmeans(frames_numpy)
        labels=self.kmeans.labels_  
        centroids=self.kmeans.cluster_centers_

        residu=frames_numpy-centroids[labels]

        i=0
        residu=initial_frames
        while error_rate >=0.001:

            residu-=residu
            self.codebook[f'codebook_{i}']=residu
            i+=1


###YOUTUBE VERSION