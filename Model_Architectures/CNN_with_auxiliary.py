import torch 
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

#Siamese CNN,  with auxiliary loss 
class CNN_aux(nn.Module):
    
    def __init__(self):
        super(CNN_aux, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 1)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(20,1)
     
        self.dropout = nn.Dropout(0.15)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):   
          
        a = x[:,0,:,:].reshape((x.size()[0],1,14,14))
        b = x[:,1,:,:].reshape((x.size()[0],1,14,14))

        a = F.max_pool2d(self.conv1(a), kernel_size = 3, stride = 3 )
        a = F.max_pool2d(self.conv2(a), kernel_size = 2, stride = 2 )
        a = self.conv3(a)
        #Fully connected layers
        a = F.selu(self.fc1(a.flatten(start_dim=1)))
        a = self.dropout(a)
        a = F.selu(self.fc2(a.flatten(start_dim=1)))
        a = self.dropout(a)
        a = self.fc3(a.flatten(start_dim=1))
        
        b = F.max_pool2d(self.conv1(b), kernel_size = 3, stride = 3 )
        b = F.max_pool2d(self.conv2(b), kernel_size = 2, stride = 2 )
        b = self.conv3(b)
        #Fully connected layers
        b = F.selu(self.fc1(b.flatten(start_dim=1)))
        b = self.dropout(b)
        b = F.selu(self.fc2(b.flatten(start_dim=1)))
        b = self.dropout(b)
        b = self.fc3(b.flatten(start_dim=1))
        
        c = torch.cat([a, b], dim=1)
        c = self.activation(self.fc4(c))
    
        return  c.squeeze(), a , b
    
    def number_parameters(self) : 
        count = 0 
        for parameter in self.parameters() :
            if parameter.requires_grad == True : 
                count += parameter.numel()
        return count
