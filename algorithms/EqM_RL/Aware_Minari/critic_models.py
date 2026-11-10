from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 
from utils.networks import MLP, QCriticMLP,VCriticMLP



class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


#################################################################################################
#################################################################################################
#################################################################################################

class DeterministicCritic(Module):
    def __init__(self,state_dim, action_dim, hidden_dim, num_hidden_layers):
        super().__init__()

        self.state_dim         = state_dim 
        self.action_dim        = action_dim
        self.hidden_dim        = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim        = 1

        self.input_dim = self.state_dim + self.action_dim

        self.model = MLP(
            input_dim         = self.input_dim, 
            output_dim        = self.output_dim, 
            hidden_dim        = self.hidden_dim,
            num_hidden_layers = self.num_hidden_layers
            )

    def forward(self,state: torch.tensor , action: torch.tensor) -> torch.Tensor:

        x = torch.cat([state,action],dim=-1)

        value = self.model(x)

        return value




class QCritic(Module):
    def __init__(self,state_dim, action_dim, hidden_dim, num_hidden_layers, num_critics):
        super().__init__()

        self.state_dim         = state_dim 
        self.action_dim        = action_dim
        self.hidden_dim        = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_critic        = num_critics
        self.output_dim        = 1

        self.input_dim = self.state_dim + self.action_dim

        self.model = QCriticMLP(
            input_dim         = self.input_dim, 
            output_dim        = self.output_dim, 
            hidden_dim        = self.hidden_dim,
            num_hidden_layers = self.num_hidden_layers
            ) 

    def forward(self,state: torch.tensor , action: torch.tensor) -> torch.Tensor:
        
        value = self.model(state,action)
        return value



class TwinCritic(Module):
    def __init__(self,state_dim, action_dim, hidden_dim, num_hidden_layers, num_critics):
        super().__init__()

        self.state_dim         = state_dim 
        self.action_dim        = action_dim
        self.hidden_dim        = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_critic        = num_critics
        self.output_dim        = 1

        self.input_dim = self.state_dim + self.action_dim

        self.q1 = QCriticMLP(
            input_dim         = self.input_dim, 
            output_dim        = self.output_dim, 
            hidden_dim        = self.hidden_dim,
            num_hidden_layers = self.num_hidden_layers
            )
        self.q2 = QCriticMLP(
            input_dim         = self.input_dim, 
            output_dim        = self.output_dim, 
            hidden_dim        = self.hidden_dim,
            num_hidden_layers = self.num_hidden_layers
            )

    def forward(self,state: torch.tensor , action: torch.tensor) -> torch.Tensor:
        
        x = self.model(state,action)
        q1_value = self.q1(x)
        q2_value = self.q2(x)     

        return q1_value,q2_value


class VCritic(Module):
    def __init__(self,state_dim, hidden_dim, num_hidden_layers, num_critics):
        super().__init__()

        self.state_dim         = state_dim 
        self.hidden_dim        = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_critic        = num_critics
        self.output_dim        = 1

        self.input_dim = self.state_dim

        self.model =  VCriticMLP(
            input_dim         = self.input_dim, 
            output_dim        = self.output_dim, 
            hidden_dim        = self.hidden_dim,
            num_hidden_layers = self.num_hidden_layers
            )

    def forward(self,state: torch.tensor ) -> torch.Tensor:
        
        value = self.model(state)
        return value













