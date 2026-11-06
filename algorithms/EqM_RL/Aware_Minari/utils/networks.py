import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Linear as lin
import math


class MLP(Module):
    def __init__(self, input_dim, output_dim,hidden_dim,num_hidden_layers):
        super().__init__()

        assert num_hidden_layers >= 1, "For MLP, num_hidden_layers must be at least 1. This class does not support 0 hidden layers."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        
        first_layer = lin(self.input_dim,self.hidden_dim)
        layers.append(first_layer)
        layers.append(nn.Mish())
        
        for _ in range(0,self.num_hidden_layers-1):
            body = lin(self.hidden_dim,self.hidden_dim)
            layers.append(body)
            layers.append(nn.Mish())
        
        head = lin(self.hidden_dim,self.output_dim)
        layers.append(head)
        self.model = nn.Sequential(*layers)

    
    def forward(self,x):
        o = self.model(x)
        return o
        


class CriticMLP(Module):
    def __init__(self, input_dim, output_dim,hidden_dim,num_hidden_layers):
        super().__init__()

        assert num_hidden_layers >= 1, "For ConditionalMLP, num_hidden_layers must be at least 1. This class does not support 0 hidden layers."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        
        first_layer = lin(self.input_dim,self.hidden_dim)
        layers.append(first_layer)
        layers.append(nn.Mish())
        layers.append(nn.LayerNorm(normalized_shape=self.hidden_dim))

        for _ in range(0,self.num_hidden_layers-1):
            body = lin(self.hidden_dim,self.hidden_dim)
            layers.append(body)
            layers.append(nn.Mish())
            layers.append(nn.LayerNorm(normalized_shape=self.hidden_dim))
        
        head = lin(self.hidden_dim,self.output_dim)
        layers.append(head)
        self.model = nn.Sequential(*layers)

    
    def forward(self,state,action):
        x = torch.concat([state,action],dim=-1)
        o = self.model(x)
        return o
        
    

        
        