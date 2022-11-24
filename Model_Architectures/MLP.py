import torch 
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

#MLP
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(392, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,1)
     
        self.dropout = nn.Dropout(0.15)
        self.activation = nn.Sigmoid() 

    def forward(self, x):   
        #Fully connected layers
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