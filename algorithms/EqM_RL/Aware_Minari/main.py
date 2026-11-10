import os
import uuid
import gymnasium as gym
import hydra
import numpy as np
import torch
from buffer import ReplayBuffer
from minari_utils import get_ref_scores, minari_normalized_score, prepare_minari_data
from actor_models import Actor
from critic_models import TwinCritic, VCritic
from agents import Otter
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from utils.networks import ConditionalMLP
from logger import global_logger as gl
import wandb
from utils.helper import set_seed,compute_mean_std,normalize_states,wrap_env,return_reward_range,modify_reward,wandb_init


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    if getattr(cfg, "run_id", None) is None:
        cfg.run_id = str(uuid.uuid4())
    else:
        cfg.run_id = str(cfg.run_id)

    safe_env_name = cfg.env_name.replace("/", "_")
    cfg.run_name = f"{cfg.name}-{safe_env_name}-{cfg.run_id[:8]}"

    if getattr(cfg, "checkpoints_path", None) is not None:
        cfg.actual_checkpoints_path = os.path.join(cfg.checkpoints_path, cfg.run_name)
        os.makedirs(cfg.actual_checkpoints_path, exist_ok=True)
    else:
        cfg.actual_checkpoints_path = None

    OmegaConf.set_struct(cfg, True)

    env, dataset, minari_dataset = prepare_minari_data(cfg.env_name)
    set_seed(cfg.seed, env, deterministic_torch=cfg.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ref_scores = get_ref_scores(minari_dataset)

    if getattr(cfg, "normalize_reward", False):
        modify_reward(dataset, cfg.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        cfg.buffer_size,
        cfg.device,
    )
    replay_buffer.load_minari_dataset(dataset)

    model_kwargs = {
        "input_dim" :        state_dim+action_dim,
        "output_dim":        action_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }
    
    Qcritic_kwargs = {
        "state_dim" :        state_dim,
        "action_dim":        action_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }

    Vcritic_kwargs = {
        "state_dim" :        state_dim,
        "hidden_dim":        cfg.hidden_dim,
        "num_hidden_layers": cfg.num_hidden_layers
    }

    model = ConditionalMLP(**model_kwargs)
    model.to(cfg.device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
        
    Qnets = TwinCritic(**Qcritic_kwargs)
    Vnet  = VCritic(**Vcritic_kwargs)
    Qnets.to(cfg.device)
    Vnet.to(cfg.device)
    Qnets_optimizer = torch.optim.Adam(Qnets.parameters(), lr=cfg.learning_rate)
    Vnet_optimizer  = torch.optim.Adam(Vnet.parameters(), lr=cfg.learning_rate)

    actor = Actor(
        model         = model,
        Qnets         = Qnets,
        Vnet          = Vnet,        
        action_dim    = action_dim,
        min_action    = env.action_space.low[0],
        max_action    = env.action_space.high[0],
        ebm           = cfg.actor.ebm,
        opt_type      = cfg.actor.opt_type,
        step_size     = cfg.actor.step_size, 
        num_step      = cfg.actor.num_step,
        moment        = cfg.actor.moment,
        sampler_type  = cfg.actor.sampler_type,
        ood_threshold = cfg.actor.ood_threshold,
        early_stop    = cfg.actor.early_stop
        )

    agent = Otter(
        model               = model,
        model_optimizer     = model_optimizer,
        actor               = actor,
        q_nets              = Qnets,
        q_nets_optimizer    = Qnets_optimizer,
        v_net               = Vnet,
        v_net_optimizer     = Vnet_optimizer,
        **cfg.agent
        )

    @torch.no_grad()
    def eval_actor(env: gym.Env, n_episodes: int, seed: int) -> np.ndarray:
        actor.eval()
        episode_rewards = []
        for i in range(n_episodes):
            state, _ = env.reset(seed=seed + i)
            done = False
            episode_reward = 0.0
            while not done:
                action = actor.sample_rejection(state)
                action = np.squeeze(action, axis=0)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            episode_rewards.append(episode_reward)

        actor.train()
        return np.asarray(episode_rewards)
    

    wandb_init(OmegaConf.to_container(cfg, resolve=True))

    if getattr(cfg, "actual_checkpoints_path", None) is not None:
        print(f"Checkpoints path: {cfg.actual_checkpoints_path}")
        with open(os.path.join(cfg.actual_checkpoints_path, "config.yaml"), "w") as f:
            # Save the fully resolved Hydra config
            OmegaConf.save(config=cfg, f=f)
    try:
        for t in trange(cfg.num_train_ops, ncols=80):
            batch = replay_buffer.sample(cfg.batch_size)
            batch = [b.to(cfg.device) for b in batch]
            update_result = agent.update(batch)
            for key, val in update_result.items():
                gl.log(f"train/{key}", val)
            gl.dump(step=t)
            if (t) % cfg.eval_frequency == 0:
                eval_scores = eval_actor(env, cfg.n_test_episodes, cfg.test_seed)

                gl.log("eval/score", eval_scores.mean())
                normalized_eval_scores = minari_normalized_score(
                    eval_scores.mean(), ref_scores
                )
                gl.log("eval/normalized_score", normalized_eval_scores)
                
                gl.dump(step=t)

                if getattr(cfg, "actual_checkpoints_path", None) is not None:
                    torch.save(
                        agent.state_dict(),
                        os.path.join(cfg.actual_checkpoints_path, f"checkpoint_{t}.pt"),
                    )
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
