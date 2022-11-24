import torch 
from torch import nn
from torch.nn import functional as F
import torch.optim as optim


# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Convolutional Layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 1)
        
        #Hidden Layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,1)
     
        self.dropout = nn.Dropout(0.15)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
       
        x = F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 3 )
        x = F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2 )
    
        x = self.conv3(x)
        
        x = F.selu(self.fc1(x.flatten(start_dim=1)))
        x = self.dropout(x)
        x = F.selu(self.fc2(x.flatten(start_dim=1)))
        x = self.dropout(x)
        x = F.selu(self.fc3(x.flatten(start_dim=1)))
        x = self.dropout(x)
        x = self.fc4(x.flatten(start_dim=1))
        x = self.activation(x)
        
        return x.squeeze()
    
    def number_parameters(self) : 
        count = 0 
        for parameter in self.parameters() :
            if parameter.requires_grad == True : 
                count += parameter.numel()
        return count