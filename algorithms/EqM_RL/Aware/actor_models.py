from typing import Tuple
from utils.helper import (
    StateNormalizer,
    flatten_repeated_states,
    unflatten_repeated_tensor,
    select_lowest_ood_actions,
)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 
from collections import deque

class Actor(nn.Module):
    def __init__(self, model, q_nets, v_net, action_dim, min_action, max_action, ebm,
                 opt_type, step_size, num_step, moment, sampler_type, ood_threshold, early_stop,
                 state_mean=0.0, state_std=1.0):
        super().__init__()
        
        self.model = model
        self.q_nets = q_nets
        self.v_net = v_net
        self.state_normalizer = StateNormalizer(state_mean, state_std)
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.opt_type = opt_type
        self.step_size = step_size
        self.num_step = num_step
        self.moment = moment
        self.ebm = ebm
        self.sampler_type = sampler_type
        self.ood_threshold = ood_threshold
        self.early_stop = early_stop
        self.descend_fn = self._descend
        self._compiled = False 
        
        
    def compile_sampler(self, mode: str = "reduce-overhead"):
        self.descend_fn = torch.compile(self._descend, mode=mode)
        self._compiled = True
        return self


    def forward(self, state: torch.Tensor, num_actions: int = 1):

        """
           Generate Actions for training.
        """
        
        predicted_actions, _ = self._sample_actions_for_training(state , num_actions)

        predicted_actions = torch.clamp(predicted_actions, min=self.min_action, max=self.max_action)

        return predicted_actions
        


    def _sample_actions_for_training(self, state_tensor: torch.Tensor, num_actions: int = 1):
        
            assert num_actions >= 1, "Number of candidate actions must be greater than 0."

            state_tensor = self.state_normalizer.normalize(state_tensor)
            device = state_tensor.device

            
            if self.sampler_type == 'implicit_OOD':
                num_samples = num_actions * 2
            else:
                num_samples = num_actions

            state_repeated = state_tensor.unsqueeze(0).repeat(num_samples, 1, 1)
            prepared_state = flatten_repeated_states(state_repeated)

            batch_size_total = prepared_state.shape[0]
            x = torch.randn((batch_size_total, self.action_dim), device=device)

            ood_scores = None

            if self.sampler_type == 'implicit':
                if self.early_stop is not None:
                    flat_all_actions = self._implicit_stop(x=x, state=prepared_state, tau_opt=self.early_stop)
                else:
                    flat_all_actions = self._implicit(x=x, state=prepared_state)

                predicted_actions = unflatten_repeated_tensor(flat_all_actions, num_samples)

            elif self.sampler_type == 'implicit_OOD':
              
                if self.early_stop is not None:
                    flat_all_actions, flat_ood_scores = self._implicit_OOD_stop(x=x, state=prepared_state, tau_opt=self.early_stop)
                else:
                    flat_all_actions, flat_ood_scores = self._implicit_OOD_leaky_bucket(x=x, state=prepared_state)

                ood_scores = unflatten_repeated_tensor(flat_ood_scores, num_samples)
                all_actions = unflatten_repeated_tensor(flat_all_actions, num_samples)

                
                predicted_actions = select_lowest_ood_actions(all_actions, ood_scores)

            else:
                raise ValueError(f"Sampler must be 'implicit' or 'implicit_OOD', got '{self.sampler_type}'")

            return predicted_actions, ood_scores
        

    @torch.no_grad()
    def sample(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluation method. Expected to be called by Gym/Gymnasium.

        Generates exactly one action per state using _implicit.
        No multiple action sampling.
        No OOD score.
        No reject-and-rank.
        No critic evaluation.
        """
        actor_was_training = self.training
        model_was_training = self.model.training

        try:
            self.eval()
            self.model.eval()

            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)

            device = next(self.model.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            state_tensor = self.state_normalizer.normalize(state_tensor)

            batch_size = state_tensor.shape[0]

            # One random initial action per state
            x = torch.randn(
                (batch_size, self.action_dim),
                device=device,
                dtype=torch.float32,
            )

            # Use only the implicit sampler
            action_tensor = self._implicit(x=x, state=state_tensor)

            action_tensor = torch.clamp(
                action_tensor,
                min=self.min_action,
                max=self.max_action,
            )

            return action_tensor.cpu().numpy()

        finally:
            self.train(actor_was_training)
            self.model.train(model_was_training)


    
    @torch.no_grad()
    def sample_rejection(self, state: np.ndarray, num_actions: int = 32,
                         generator=None, return_info: bool = False):

        single_state = state.ndim == 1

        if single_state:
            state = np.expand_dims(state, axis=0)

        device = next(self.model.parameters()).device

        raw_states = torch.tensor(state, dtype=torch.float32, device=device)
        batch_size, state_dim = raw_states.shape

        normalized_states = self.state_normalizer.normalize(raw_states)
        repeated_normalized_states = normalized_states.repeat_interleave(num_actions, dim=0)
        initial_actions = torch.randn(
            batch_size * num_actions,
            self.action_dim,
            device=device,
            generator=generator,
        )
        candidate_actions = self._implicit(x=initial_actions, state=repeated_normalized_states)
        candidate_actions = torch.clamp(
            candidate_actions,
            min=self.min_action,
            max=self.max_action,
        )
        candidate_actions = candidate_actions.view(batch_size, num_actions, self.action_dim)


        critic_states = raw_states.unsqueeze(dim=1)
        critic_states = critic_states.expand(batch_size, num_actions, state_dim)
        q1_values, q2_values = self.q_nets(critic_states, candidate_actions)
        q_values = torch.min(q1_values, q2_values)
        q_values = q_values.squeeze(dim=-1)

        best_q_values, best_action_indices = q_values.max(dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        best_actions = candidate_actions[batch_indices, best_action_indices]
        actions = best_actions.cpu().numpy()

        if single_state:
            actions = actions[0]
        if not return_info:
            return actions

        mean_candidate_q = q_values.mean(dim=1)
        selection_gains = best_q_values - mean_candidate_q
        info = {
            "q_start": best_q_values.mean().item(),
            "gain": selection_gains.sum().item(),
            "n": batch_size,
        }
        return actions, info


    def _implicit(self, x , state):

        is_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            
            x = self.descend_fn(x,state)
            if self._compiled:
                x = x.clone()
        
        if is_training:
            self.model.train()
        return x
    
    
    def _descend(self,x,state):
        if self.opt_type == "gd":
            for _ in range(self.num_step):
                grad = self.model(x,state)
                x = x - self.step_size * grad

        elif self.opt_type == "nag":
            m = torch.zeros_like(x)
            for _ in range(self.num_step):
                x_lookahead = x - self.step_size * m * self.moment
                grad = self.model(x_lookahead,state)
                m = grad
                x = x - self.step_size * m
        else:
            raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
        return x
                
        
    

    def _implicit_OOD(self, x , state):
            
            is_training = self.model.training
            self.model.eval()
    
            with torch.no_grad():
                if self.opt_type == "gd":
                    for _ in range(self.num_step):
                        grad = self.model(x,state)
                        x = x - self.step_size * grad
    
                elif self.opt_type == "nag":
                    m = torch.zeros_like(x)
                    for _ in range(self.num_step):
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)
                        m = grad
                        x = x - self.step_size * m
                else:
                    raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
              

                final_grad = self.model(x,state)
                
                
                ood_scores = torch.norm(final_grad, p=2, dim=1, keepdim=True)
            

            if is_training:
                self.model.train()
            return x, ood_scores
    
    def _implicit_stop(self, x , state, tau_opt = 1.5):
            """
            GeCO Adaptive Early Stopping: Particles individually stop updating 
            once their gradient norm falls below tau_opt.
            """
            is_training = self.model.training
            self.model.eval()

            with torch.no_grad():
                if self.opt_type == "gd":
                    # Create a mask tracking which particles are still moving (True = Active)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        grad = self.model(x,state)

                        # Calculate the L2 norm of the gradients
                        grad_norms = torch.norm(grad, p=2, dim=1)

                        # Update active mask: keep True ONLY if norm > tau_opt AND it was previously active
                        active_mask = active_mask & (grad_norms > tau_opt)

                        # If all particles have stopped, exit the loop completely
                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        # Apply update only to active particles (multiply grad by active_mask)
                        # active_mask.unsqueeze(1) broadcasts the 1D mask to the 2D coordinates [batch, 2]
                        grad_update = grad * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * grad_update

                elif self.opt_type == "nag":
                    m = torch.zeros_like(x)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        # Lookahead
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)

                        grad_norms = torch.norm(grad, p=2, dim=1)
                        active_mask = active_mask & (grad_norms > tau_opt)

                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        m = grad
                        # Apply update only to active particles
                        m_update = m * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * m_update
                else:
                    raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")


            if is_training:
                self.model.train()
            return x
    

    def _implicit_OOD_stop(self, x , state, tau_opt=0.4):
            """
            Combined GeCO Sampler: 
            1. Adaptive Early Stopping (particles park when grad_norm < tau_opt)
            2. OOD Detection (returns the final gradient norms for anomaly filtering)
            """

            is_training = self.model.training
            self.model.eval()

            with torch.no_grad():
                if self.opt_type == "gd":
                    # Create a mask tracking which particles are still moving (True = Active)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        grad = self.model(x,state)

                        # Calculate the L2 norm of the gradients
                        grad_norms = torch.norm(grad, p=2, dim=1)

                        # Update active mask: keep True ONLY if norm > tau_opt AND it was previously active
                        active_mask = active_mask & (grad_norms > tau_opt)

                        # If all particles have stopped, exit the loop completely
                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        # Apply update only to active particles (multiply grad by active_mask)
                        grad_update = grad * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * grad_update

                elif self.opt_type == "nag":
                    m = torch.zeros_like(x)
                    active_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

                    for step in range(self.num_step):
                        # Lookahead
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)

                        grad_norms = torch.norm(grad, p=2, dim=1)
                        active_mask = active_mask & (grad_norms > tau_opt)

                        if not active_mask.any():
                            print(f"GeCO Early Stopping: All particles converged at step {step}.")
                            break

                        m = grad
                        # Apply update only to active particles
                        m_update = m * active_mask.unsqueeze(1).float()
                        x = x - self.step_size * m_update
                else:
                    raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")

                final_grad = self.model(x,state)
                ood_scores = torch.norm(final_grad, p=2, dim=1, keepdim=True)

            if is_training:
                self.model.train()
            return x, ood_scores
        
    def _implicit_OOD_moving_avg(self, x , state):
            
            is_training = self.model.training
            self.model.eval()
    
            with torch.no_grad():
                if self.opt_type == "nag":
                    m = torch.zeros_like(x)
                    window_length = 5
                    score_window = deque(maxlen=window_length)
                    
                    for _ in range(self.num_step):
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)
                        m = grad 
                        ood_moving_avg_score, score_window = self._ood_moving_average(grad, score_window)
                        x = x - self.step_size * m
                else:
                    raise ValueError(f"\n Action Gradient optimizer must be 'nag', got '{self.opt_type}' \n ")
              

            if is_training:
                self.model.train()
            return x, ood_moving_avg_score

    def _implicit_OOD_leaky_bucket(self, x, state):
        is_training = self.model.training
        self.model.eval()

        tau = 0.5
        lambda_decay = 0.0


        with torch.no_grad():
            if self.opt_type == "nag":
                m = torch.zeros_like(x)
                ood_score_bucket = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)
                
                for _ in range(self.num_step):
                    x_lookahead = x - self.step_size * m * self.moment
                    grad = self.model(x_lookahead, state)
                    m = grad 
                    ood_score_bucket = self._ood_leaky_bucket(grad, ood_score_bucket, tau, lambda_decay)

                    x = x - self.step_size * m
            else:
                raise ValueError(f"\n Action Gradient optimizer must be 'nag', got '{self.opt_type}' \n ")
          
        if is_training:
            self.model.train()
            
        return x, ood_score_bucket

    def _implicit_langevin(self, x , state , initial_temperature=1.0, noise_decay=0.99):

            is_training = self.model.training
            self.model.eval()

            with torch.no_grad():
                if self.opt_type == "gd":
                    for i in range(self.num_step):
                       
                        grad = self.model(x,state)

                       
                        current_temp = initial_temperature * (noise_decay ** i)

                        
                        noise_scale = np.sqrt(2 * self.step_size * current_temp)
                        noise = torch.randn_like(x)

                        
                        x = x - self.step_size * grad + noise_scale * noise

                elif self.opt_type == "nag":
                    m = torch.zeros_like(x)
                    for i in range(self.num_step):
                       
                        x_lookahead = x - self.step_size * m * self.moment
                        grad = self.model(x_lookahead,state)
                        m = grad

                       
                        current_temp = initial_temperature * (noise_decay ** i)

                  
                        noise_scale = np.sqrt(2 * self.step_size * current_temp)
                        noise = torch.randn_like(x)

                        x = x - self.step_size * m + noise_scale * noise
                else:
                    raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
            
            if is_training:
                self.model.train()
            return x
        
        
    def _compute_gradient(self, Xt, state):
        # We assume Xt already has requires_grad_(True) when passed in
        output = self.model(Xt,state)
            
        if self.ebm == 'dot':
            E = torch.sum(output * Xt, dim=1)     
        elif self.ebm == 'scalar':
            E = output.squeeze(-1)       
        elif self.ebm == 'l2':
            E = -0.5 * torch.sum(output**2, dim=1)        
        else:
            raise ValueError("\n During Sampling - Type of EBM set incorrectly; chose one from: l2 , scalar , dot \n ") 

        pred_grad = torch.autograd.grad(
            outputs=[E.sum()],
            inputs=[Xt],
            create_graph=False
        )[0]
        
        return pred_grad 
    

    def _explicit(self, x , state):

        is_training = self.model.training
        self.model.eval()
        with torch.enable_grad():     
            if self.opt_type == "gd":        
                # Start with a clean, detached tensor
                x = x.detach()

                for _ in range(self.num_step):
                    # 1. Turn on tracking for this specific step
                    x.requires_grad_(True)

                    # 2. Get the gradient
                    grad = self._compute_gradient(x,state)

                    # 3. Update the state and immediately DETACH to prevent a memory leak
                    x = (x - self.step_size * grad).detach()

            elif self.opt_type == "nag":     
                x = x.detach()
                m = torch.zeros_like(x)

                for _ in range(self.num_step):
                    # 1. Calculate lookahead and detach it so it's a clean starting point
                    x_lookahead = (x - self.step_size * m * self.moment).detach()

                    # 2. Turn on tracking for the lookahead position
                    x_lookahead.requires_grad_(True)

                    # 3. Get the gradient
                    grad = self._compute_gradient(x_lookahead,state)

                    # 4. Save momentum (detached to avoid tracking history)
                    m = grad.detach()

                    # 5. Update the main state and detach
                    x = (x - self.step_size * m).detach()
            else:
                raise ValueError(f"\n Action Gradient optimizer must be 'gd' or 'nag', got '{self.opt_type}' \n ")
        
        if is_training:
            self.model.train()
        return x


    def _ood_moving_average(self, grad, score_window):
        score = torch.linalg.norm(grad, dim=1, keepdim=True)
        score_window.append(score)
        score_mean = torch.stack(list(score_window), dim=0).mean(dim=0)
        return score_mean, score_window


    def _ood_leaky_bucket(self, grad, ood_score_bucket, tau=0.5, lambda_decay=0.0):
        score = torch.linalg.norm(grad, dim=1, keepdim=True)
        excess = torch.relu(score - tau)
        ood_score_bucket = torch.relu((1.0 - lambda_decay) * ood_score_bucket + excess)
        return ood_score_bucket
