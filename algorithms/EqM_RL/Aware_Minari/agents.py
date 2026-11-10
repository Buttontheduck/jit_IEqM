from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import global_logger as gl

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def expand_t_like_x(t, x):
    
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t
  

class Otter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        model_optimizer: torch.optim.Optimizer,
        actor: nn.Module,
        q_nets: nn.Module,
        q_nets_optimizer: torch.optim.Optimizer,
        v_net: nn.Module,
        v_net_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        temperature: float = 1.0,
        exp_adv_max: float = 100.0,
        fino: bool = False,
        flag_type: str = 'trun',
        flag_threshold: float = 0.8,
        scale_gradient: float = 4.0,
        num_actions: int = 10,
        fino_knob: float = 0.1,
        **kwargs 
        
    ):
        super().__init__()
        self._model = model
        self._model_optimizer = model_optimizer
        
        self._actor = actor

        self._q_nets = q_nets
        self._q_nets_optimizer = q_nets_optimizer
        self._target_q_nets = deepcopy(q_nets)

        self._v_net = v_net
        self._v_net_optimizer = v_net_optimizer

        self._gamma = gamma
        self._tau = tau
        self._temperature = temperature
        self._exp_adv_max = exp_adv_max
        
        self.fino = fino
        self.flag_type = flag_type
        self.flag_threshold = flag_threshold
        self.scale_gradient = scale_gradient
        self.fino_knob = fino_knob

        self.num_actions = num_actions
        
        
        

    def _expand_dims(self,low_dim_vector,reference):
        vector_matched_dim = expand_t_like_x(low_dim_vector,reference)
        return vector_matched_dim
    
    def _c_linear(self,rho):
        return 1-rho
    
    def _c_trun(self,interpolate,rho):
        

        if not (0 < interpolate < 1):
              raise ValueError(f"Interpolate must be in (0,1), got {interpolate}")

        start = 1.0
        ct = torch.minimum(start-(start-1)/(interpolate)*rho, 1/(1-interpolate)-1/(1-interpolate)*rho)
        return ct

    def _sample_gamma_and_noise(self,true_action):
        
        batch_size = true_action.shape[0]
        device = true_action.device
        dtype = true_action.dtype

        rho = torch.rand(batch_size, device=device, dtype=dtype)  
        noise = torch.randn_like(true_action)
        return rho, noise
    
    def _mix_data_and_compute_target(self,rho,noise,true_action):
    
        rho = self._expand_dims(rho,true_action)
        noisy_action = rho * (true_action) + (1 - rho) * noise
        target_gradient = noise-true_action
        return noisy_action, target_gradient
    
    def _fino_mix_data_and_compute_target(self, rho, noise, true_action, eta=0.1):
           
            """
            FINO Noise Injection: Expands the learned action space by injecting 
            a scheduled Gaussian noise and modifying the target vector.
            """
            rho = self._expand_dims(rho, true_action)

            # 1. Standard interpolation
            noisy_action = rho * true_action + (1 - rho) * noise

            # 2. FINO Variance Schedule: alpha^2 = (eta^2 - 2*eta)*rho^2 + 2*eta*rho
            alpha_sq = (eta**2 - 2*eta) * (rho**2) + 2 * eta * rho

            # Clamp at 0 for numerical safety before taking the square root
            alpha = torch.sqrt(torch.clamp(alpha_sq, min=0.0)) 

            # 3. Inject the FINO noise into the state
            fino_noise = torch.randn_like(true_action)
            noisy_action_fino = noisy_action + (alpha * fino_noise)

            # 4. Modified FINO Target Gradient
            # Standard EqM is (noise - true_action). FINO scales the noise weight by (1-eta).
            target_gradient = (1 - eta) * noise - true_action

            return noisy_action_fino, target_gradient

    def _bc_actor_loss(self, states, actions):
        
        """
        B: Batch  Dimensions
        A: Action Dimensions
        S: State  Dimensions
        K: Number of Samples
        """
        
        with torch.no_grad():

            rho, noise = self._sample_gamma_and_noise(actions)
                            
            if self.flag_type=="trun":
                ct = self._c_trun(interpolate=self.flag_threshold,rho=rho)
            elif self.flag_type=="lin":
                ct = self._c_linear(rho=rho)
            else:
                raise ValueError(f"\n Invalid Ct FlagType: {self.flag_type!r}. ""Expected either 'lin' or 'trun'. \n")

            ct = self._expand_dims(low_dim_vector=ct,reference=actions)

            
            if self.fino:
                noisy_action, target_gradient =  self._fino_mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions,eta=self.fino_knob)
            else:
                noisy_action, target_gradient =  self._mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions)
            
            scaled_target_grad =  self.scale_gradient * ct * target_gradient
            
            
        pred_grad = self._model(noisy_action,states)

        loss_per_sample_and_dim = F.mse_loss(pred_grad, scaled_target_grad, reduction='none') # Returns (B,A), loss for each sample and each action dimension 
        loss_per_sample = loss_per_sample_and_dim.sum(dim=-1, keepdim=True) # Returns (B,1), loss for each gradient prediction
        
        loss= loss_per_sample.mean() 
        
        return loss
    
    def _rl_actor_loss(self, states, actions):
        
        """
        B: Batch  Dimensions
        A: Action Dimensions
        S: State  Dimensions
        K: Number of Samples
        """
        
        with torch.no_grad():

            rho, noise = self._sample_gamma_and_noise(actions)
                            
            if self.flag_type=="trun":
                ct = self._c_trun(interpolate=self.flag_threshold,rho=rho)
            elif self.flag_type=="lin":
                ct = self._c_linear(rho=rho)
            else:
                raise ValueError(f"\n Invalid Ct FlagType: {self.flag_type!r}. ""Expected either 'lin' or 'trun'. \n")

            ct = self._expand_dims(low_dim_vector=ct,reference=actions)

            
            if self.fino:
                noisy_action, target_gradient =  self._fino_mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions,eta=self.fino_knob)
            else:
                noisy_action, target_gradient =  self._mix_data_and_compute_target(rho=rho,noise=noise,true_action=actions)
            
            scaled_target_grad =  self.scale_gradient * ct * target_gradient
            
            
        pred_grad = self._model(noisy_action,states)
        
        weights = self._offline_weights(states=states, actions= actions, num_actions=self.num_actions)

        loss_per_sample_and_dim = F.mse_loss(pred_grad, scaled_target_grad, reduction='none') # Returns (B,A), loss for each sample and each action dimension 
        loss_per_sample = loss_per_sample_and_dim.sum(dim=-1, keepdim=True) # Returns (B,1), loss for each gradient prediction
        weighted_loss_per_sample = loss_per_sample * weights # Returns (B,1)
        loss = weighted_loss_per_sample.mean() # Scalar Loss
        
        #loss= loss_per_sample.mean() # scalar loss for BC case.
        
        return loss

    
    def _offline_weights(self, states, actions):
            with torch.no_grad():

                q1_value, q2_value = self._q_nets(states, actions)
                q = torch.min(q1_value, q2_value)
                v = self._v_net(states) # shape: [B, 1]

                adv = q - v
                
                

                weights = torch.clamp_max(input=torch.exp(adv/self._temperature), max=self._exp_adv_max) 
                gl.log("weighting/q_mean",            q.mean().item())
                gl.log("weighting/v_mean",            v.mean().item())
                gl.log("weighting/adv_mean",          adv.mean().item())
                gl.log("weighting/adv_std",           adv.std(unbiased=False).item())
                gl.log("weighting/weights_mean",      weights.mean().item())
                gl.log("weighting/weights_max",       weights.max().item())
                gl.log("weighting/weights_min",       weights.min().item())
                gl.log("weighting/weights_var",       weights.var(unbiased=False).item())
                gl.log("weighting/weights_median",    weights.median().item())
                gl.log("weighting/weights_clip_frac", (weights >= self._exp_adv_max).float().mean().item())
                
                
                
            return weights
    

    def _v_critic_loss(self, states, actions):
        
        with torch.no_grad():
            q_1,q_2 = self._q_nets(states,actions)
            v = self._v_net(states)
            q = torch.min(q_1,q_2)
            d = q - v
            if d>=0:
                loss = torch.multiply(0.9,d**2)
            else:
                loss = torch.multiply(0.1,d**2)
        return loss
    
    def _q_critic_loss(self, states, actions, rewards, dones):
        
        with torch.no_grad():
            v_next = self._v_net(states)
            target = rewards + self._gamma * dones * v_next
            q_1,q_2 = self._q_nets(states,actions)
            q1_diff = (q_1 - target)**2
            q2_diff = (q_2 - target)**2
        
        loss = q1_diff + q2_diff
        return loss


    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss_q = self._q_critic_loss(states, actions, rewards, dones)
        loss_v = self._v_critic_loss(states, actions)
        self._q_nets_optimizer.zero_grad()
        self._v_net_optimizer.zero_grad()
        loss_q.backward()
        loss_v.backward()
        self._q_nets_optimizer.step()
        self._v_net_optimizer.step()
        return loss_q.item(), loss_v.item()

    def _update_actor(self, states, actions):
        loss = self._bc_actor_loss(states, actions)
        self._model_optimizer.zero_grad()
        loss.backward()
        self._model_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, truncations, terminals = batch
        dones = terminals
        q_loss,v_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss  = self._update_actor(states, actions)
        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)
        #result = {"actor_loss": actor_loss}
        result = {"q_critic_loss": q_loss,"v_critic_loss": v_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self._model.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._model.load_state_dict(state_dict["model"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"]) 





###################### CODE PARTS FROM AWAC ############################

    # def _actor_loss(self, states, actions):
    #     with torch.no_grad():
    #         pi_action, _ = self._model(states)
    #         v = torch.min(
    #             self._critic_1(states, pi_action), self._critic_2(states, pi_action)
    #         )

    #         q = torch.min(
    #             self._critic_1(states, actions), self._critic_2(states, actions)
    #         )
    #         adv = q - v
    #         weights = torch.clamp_max(
    #             torch.exp(adv / self._awac_lambda), self._exp_adv_max
    #         )

    #     action_log_prob = self._model.log_prob(states, actions)
    #     loss = (-action_log_prob * weights).mean()
    #     return loss
