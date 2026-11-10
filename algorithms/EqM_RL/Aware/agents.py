from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import global_logger as gl
from utils.helper import normalize_state_tensor

TensorBatch = List[torch.Tensor]


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), tau)


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
        v_net: nn.Module,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        batch_size: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        expectile: float = 0.9,
        temperature: float = 1.0,
        exp_adv_max: float = 100.0,
        fino: bool = False,
        flag_type: str = 'trun',
        flag_threshold: float = 0.8,
        scale_gradient: float = 8.0,
        num_actions: int = 1,
        fino_knob: float = 0.1,
    ):
        super().__init__()
        self._model = model
        self._model_optimizer = model_optimizer
        self._actor = actor
        self._q_nets = q_nets
        self._target_q_nets = deepcopy(q_nets).requires_grad_(False)
        self._v_net = v_net
        self._state_mean = state_mean
        self._state_std = state_std
        self._gamma = gamma
        self._tau = tau
        self._expectile = expectile
        self._temperature = temperature
        self._exp_adv_max = exp_adv_max
        self.fino = fino
        self.flag_type = flag_type
        self.flag_threshold = flag_threshold
        self.scale_gradient = scale_gradient
        self.fino_knob = fino_knob
        self.num_actions = num_actions
        self._ones = torch.ones(batch_size, 1, device=state_mean.device)
        self._loss_fn = self._total_loss

    def compile_losses(self, mode="reduce-overhead"):
        self._loss_fn = torch.compile(self._total_loss, mode=mode)
        return self

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
        
        loss = (loss_per_sample * self._ones).mean()
        
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
            
            
        actor_states = normalize_state_tensor(states, self._state_mean, self._state_std)
        pred_grad = self._model(noisy_action, actor_states)
        
        weights = self._offline_weights(states=states, actions=actions)

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
                gl.log("weighting/q_mean",            q.mean())
                gl.log("weighting/v_mean",            v.mean())
                gl.log("weighting/adv_mean",          adv.mean())
                gl.log("weighting/adv_std",           adv.std(unbiased=False))
                gl.log("weighting/weights_mean",      weights.mean())
                gl.log("weighting/weights_max",       weights.max())
                gl.log("weighting/weights_min",       weights.min())
                gl.log("weighting/weights_var",       weights.var(unbiased=False))
                gl.log("weighting/weights_median",    weights.median())
                gl.log("weighting/weights_clip_frac", (weights >= self._exp_adv_max).float().mean())
                
                
                
            return weights
    
    def _v_critic_loss(self, states, actions):
        with torch.no_grad():
            q_1, q_2 = self._target_q_nets(states, actions)
            q = torch.min(q_1, q_2)
        v = self._v_net(states)
        d = q - v
        weights = torch.where(d >= 0, self._expectile, 1.0 - self._expectile)
        loss = (weights * d.square()).mean()
        return loss, v, d
    
    def _q_critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            v_next = self._v_net(next_states)
            target = rewards + self._gamma * (1.0 - dones) * v_next
        q_1, q_2 = self._q_nets(states, actions)
        q1_diff = (q_1 - target).square()
        q2_diff = (q_2 - target).square()
        loss = (q1_diff + q2_diff).mean()
        return loss, q_1, q_2, target

    def _total_loss(self, states, actions, rewards, dones, next_states):
        v_loss, v, d = self._v_critic_loss(states, actions)
        q_loss, q_1, q_2, target = self._q_critic_loss(states, actions, rewards, dones, next_states)
        actor_states = normalize_state_tensor(states, self._state_mean, self._state_std)
        actor_loss = self._bc_actor_loss(actor_states, actions)
        return v_loss + q_loss + actor_loss, v_loss, q_loss, actor_loss, v, d, q_1, q_2, target

    def update(self, batch: TensorBatch, with_stats=True) -> Dict[str, torch.Tensor]:
        states, actions, rewards, next_states, truncations, terminals = batch
        loss, v_loss, q_loss, actor_loss, v, d, q_1, q_2, target = self._loss_fn(
            states, actions, rewards, terminals, next_states,
        )
        self._model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # IEqM updates the target from Q's parameters before the optimizer step.
        soft_update(self._target_q_nets, self._q_nets, self._tau)
        self._model_optimizer.step()
        if not with_stats:
            return None
        with torch.no_grad():
            q = torch.min(q_1, q_2)
            result = {
                "q_critic_loss": q_loss.detach().clone(),
                "v_critic_loss": v_loss.detach().clone(),
                "actor_loss": actor_loss.detach().clone(),
                "v_data": v.mean(),
                "q_data": q.mean(),
                "q_data_min": q.min(),
                "q_data_max": q.max(),
                "target": target.mean(),
                "adv": d.mean(),
                "adv_std": d.std(),
                "adv_pos_share": (d >= 0).float().mean(),
            }
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self._model.state_dict(),
            "q_nets": self._q_nets.state_dict(),
            "target_q_nets": self._target_q_nets.state_dict(),
            "v_net": self._v_net.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._model.load_state_dict(state_dict["model"])
        self._q_nets.load_state_dict(state_dict["q_nets"])
        self._target_q_nets.load_state_dict(state_dict["target_q_nets"])
        self._v_net.load_state_dict(state_dict["v_net"])
