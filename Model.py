import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        
        self.Conv_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.ReLU())
        
        self.Linear_layer = nn.Sequential(nn.Linear(5408, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,256),
                                     nn.ReLU(),
                                     nn.Linear(256,2),
                                     nn.Softmax(dim=1))
        
        return
    
    def forward(self, inp):
        
        out = self.Conv_layer(inp)
        #print(out.shape)
        out = out.view(-1,5408)
        out = self.Linear_layer(out)
        
        return out
    
