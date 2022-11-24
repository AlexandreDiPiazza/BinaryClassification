import torch 
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

#Siamse MLP, with auxiliary loss
class MLP_aux(nn.Module):
    

    def __init__(self):
        super(MLP_aux, self).__init__()
        # same layers as MLP without auxiliary 
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(20,1)
     
        self.dropout = nn.Dropout(0.15)
        self.activation = nn.Sigmoid() 
    
    def forward(self, x):   
          
        a = x[:,0,:,:]
        b = x[:,1,:,:]

        a = F.relu(self.fc1(a.flatten(start_dim=1)))
        a = self.dropout(a)
        a = F.relu(self.fc2(a)) 
        a = self.fc3(a) # cross entropy applies softmax so no activation for last layer
        
        b = F.relu(self.fc1(b.flatten(start_dim=1)))
        b = self.dropout(b)
        b = F.relu(self.fc2(b)) 
        b = self.fc3(b) # cross entropy applies softmax so no activation for last layer
        
        c = torch.cat([a, b], dim=1)
        c = self.activation(self.fc4(c))
        
        return  c.squeeze(), a , b
    
    def number_parameters(self) : 
        count = 0 
        for parameter in self.parameters() :
            if parameter.requires_grad == True : 
                count += parameter.numel()
        return count