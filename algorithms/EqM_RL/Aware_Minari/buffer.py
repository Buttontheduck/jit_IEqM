from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional

TensorBatch = List[torch.Tensor]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        min_action: float = -1.0,
        max_action: float = 1.0,
        action_clip_eps: float = 1e-5,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._truncations = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._terminals = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._device = device
        self._min_action = min_action
        self._max_action = max_action
        self._action_clip_eps = action_clip_eps

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_minari_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(np.clip(
            data["actions"], self._min_action + self._action_clip_eps,
            self._max_action - self._action_clip_eps,
        ))
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._truncations[:n_transitions] = self._to_tensor(
            data["truncations"][..., None]
        )
        self._terminals[:n_transitions] = self._to_tensor(
            data["terminations"][..., None]
        )
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        # d4rl.qlearning_dataset already drops timeout transitions, so truncations stay 0.
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(np.clip(
            data["actions"], self._min_action + self._action_clip_eps,
            self._max_action - self._action_clip_eps,
        ))
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._terminals[:n_transitions] = self._to_tensor(
            data["terminals"][..., None]
        )
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int, generator=None) -> TensorBatch:
        indices = torch.randint(0, min(self._size, self._pointer), (batch_size,),
                                device=self._device, generator=generator)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        truncations = self._truncations[indices]
        terminals = self._terminals[indices]
        return [states, actions, rewards, next_states, truncations, terminals]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError
