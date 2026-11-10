from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module 
from torch.nn import Linear as lin 
from torch.distributions import Normal
import torch.nn.functional as F 
import os
import random
import uuid
from typing import Optional, Tuple, Union
import wandb

import gym
import hydra
import numpy as np
import torch
import torch.nn.functional


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    def normalize_state(state):
        # Ensure the state matches the 32-bit float dtype standard in RL
        return ((state - state_mean) / state_std).astype(np.float32)

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config.get("project"),
        group=config.get("group"),
        name=config.get("run_name"),
        id=config.get("run_id"),
    )




def flatten_repeated_states(states: torch.Tensor) -> torch.Tensor:
    """
    Input:  [num_samples, batch_size, state_dim]
    Output: [batch_size * num_samples, state_dim]
    """
    return states.permute(1, 0, 2).reshape(-1, states.shape[-1])

def unflatten_repeated_tensor(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Input:  [batch_size * num_samples, dim]
    Output: [num_samples, batch_size, dim]
    """
    total, dim = tensor.shape
    assert total % num_samples == 0, f"Tensor size {total} must be divisible by {num_samples}"

    batch_size = total // num_samples
    return tensor.reshape(batch_size, num_samples, dim).permute(1, 0, 2).contiguous()

def select_lowest_ood_actions(actions: torch.Tensor, ood_scores: torch.Tensor) -> torch.Tensor:
    """
    actions:    [num_samples, batch_size, action_dim]
    ood_scores: [num_samples, batch_size, 1]
    
    Returns the actions associated with the lowest half of the OOD scores.
    """
    assert actions.ndim == 3 and ood_scores.ndim == 3
    assert actions.shape[:2] == ood_scores.shape[:2]
    assert ood_scores.shape[-1] == 1

    k = actions.shape[0] // 2


    _, indices = torch.topk(ood_scores, k=k, dim=0, largest=False, sorted=True)


    indices = indices.expand(-1, -1, actions.shape[-1])

    selected_actions = torch.gather(actions, dim=0, index=indices)
    
    return selected_actions


