import os
import time
import uuid

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from buffer import ReplayBuffer
from actor_models import Actor
from critic_models import TwinCritic, VCritic
from agents import Otter
from utils.networks import ConditionalMLP
from logger import global_logger as gl
from utils.helper import set_seed, compute_mean_std, modify_reward, wandb_init


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    from minari_utils import prepare_minari_data, get_ref_scores, minari_normalized_score

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
    ref_scores = get_ref_scores(minari_dataset)
    set_seed(cfg.seed, env, deterministic_torch=cfg.deterministic_torch)
    state_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])

    if cfg.normalize_reward:
        modify_reward(dataset, cfg.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    # Keep raw states in the buffer: only the actor uses these normalization tensors.
    state_mean = torch.tensor(state_mean, dtype=torch.float32, device=cfg.device)
    state_std = torch.tensor(state_std, dtype=torch.float32, device=cfg.device)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        cfg.buffer_size,
        cfg.device,
        min_action=low,
        max_action=high,
        action_clip_eps=cfg.action_clip_eps,
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
        "hidden_dim":        cfg.critic_hidden_dim,
        "num_hidden_layers": cfg.critic_num_hidden_layers
    }
    Vcritic_kwargs = {
        "state_dim" :        state_dim,
        "hidden_dim":        cfg.critic_hidden_dim,
        "num_hidden_layers": cfg.critic_num_hidden_layers
    }

    # Initialization order must match IEqM's random stream: V, Q1/Q2, then actor.
    Vnet = VCritic(**Vcritic_kwargs)
    Vnet.to(cfg.device)
    Qnets = TwinCritic(**Qcritic_kwargs)
    Qnets.to(cfg.device)
    model = ConditionalMLP(**model_kwargs)
    model.to(cfg.device)
    # IEqM uses this one optimizer for V, both Q networks, and the actor.
    model_optimizer = torch.optim.Adam(
        [*Vnet.parameters(), *Qnets.parameters(), *model.parameters()],
        lr=cfg.learning_rate,
        fused=cfg.device.startswith("cuda"),
    )

    actor = Actor(
        model         = model,
        q_nets        = Qnets,
        v_net         = Vnet,
        action_dim    = action_dim,
        min_action    = low,
        max_action    = high,
        ebm           = cfg.actor.ebm,
        opt_type      = cfg.actor.opt_type,
        step_size     = cfg.actor.step_size,
        num_step      = cfg.actor.num_step,
        moment        = cfg.actor.moment,
        sampler_type  = cfg.actor.sampler_type,
        ood_threshold = cfg.actor.ood_threshold,
        early_stop    = cfg.actor.early_stop,
        state_mean    = state_mean,
        state_std     = state_std
        )
    agent = Otter(
        model               = model,
        model_optimizer     = model_optimizer,
        actor               = actor,
        q_nets              = Qnets,
        v_net               = Vnet,
        state_mean          = state_mean,
        state_std           = state_std,
        batch_size          = cfg.batch_size,
        **cfg.agent
        )
    if cfg.compile:
        agent.compile_losses(mode=cfg.compile_mode)

    eval_envs = [env] + [minari_dataset.recover_environment() for _ in range(cfg.n_test_episodes - 1)]
    eval_generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 2000)

    @torch.no_grad()
    def eval_actor(envs, n_episodes: int, seed: int):
        actor.eval()
        observations = []
        for i, eval_env in enumerate(envs):
            state, _ = eval_env.reset(seed=seed + i)
            observations.append(state)
        observations = np.stack(observations).astype(np.float64)
        episode_rewards = np.zeros(n_episodes)
        episode_lengths = np.zeros(n_episodes, dtype=int)
        done = np.zeros(n_episodes, dtype=bool)
        flipped = np.zeros(n_episodes, dtype=bool)
        track_flips = "halfcheetah" in cfg.env_name
        q_start, q_gain, state_count = None, 0.0, 0

        while not done.all():
            active = np.flatnonzero(~done)
            actions, info = actor.sample_rejection(
                observations[active].astype(np.float32),
                num_actions=cfg.num_actions_inference,
                generator=eval_generator,
                return_info=True,
            )
            actions = np.clip(actions, low, high)
            if q_start is None:
                q_start = info["q_start"]
            q_gain += info["gain"]
            state_count += info["n"]
            for k, i in enumerate(active):
                observations[i], reward, terminated, truncated, _ = envs[i].step(actions[k])
                done[i] = terminated or truncated
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                if track_flips:
                    flipped[i] |= abs(observations[i][1]) > cfg.flip_angle

        actor.train()
        normalized_score = minari_normalized_score(episode_rewards.mean(), ref_scores)
        result = {
            "score": float(episode_rewards.mean()),
            "score_std": float(episode_rewards.std()),
            "score_min": float(episode_rewards.min()),
            "normalized_score": float(normalized_score),
            "episode_length": float(episode_lengths.mean()),
            "q_start": q_start,
            "q_gain": q_gain / max(state_count, 1),
        }
        if track_flips:
            result["flip_rate"] = float(flipped.mean())
        return result

    wandb_init(OmegaConf.to_container(cfg, resolve=True))
    if cfg.actual_checkpoints_path is not None:
        print(f"Checkpoints path: {cfg.actual_checkpoints_path}")
        with open(os.path.join(cfg.actual_checkpoints_path, "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f)

    batch_generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 1000)
    gl.buffer.clear()
    eval_history = []
    last_log_time, last_log_step = time.time(), 0
    try:
        for t in trange(1, cfg.num_train_ops, ncols=80):
            do_eval = t % cfg.eval_frequency == 0 or t == cfg.num_train_ops - 1
            do_log = t % cfg.log_frequency == 0 or do_eval
            batch = replay_buffer.sample(cfg.batch_size, generator=batch_generator)
            update_result = agent.update(batch, with_stats=t % cfg.stats_every == 0 or do_log)
            for key, val in (update_result or {}).items():
                gl.log(f"train/{key}", val)
            if do_log:
                gl.log("train/ms_per_step", 1000 * (time.time() - last_log_time) / (t - last_log_step))
            if do_eval:
                started = time.time()
                eval_result = eval_actor(eval_envs, cfg.n_test_episodes, cfg.test_seed)
                eval_result["seconds"] = time.time() - started
                eval_history.append(eval_result["normalized_score"])
                for key, val in eval_result.items():
                    gl.log(f"eval/{key}", val)
            if do_log:
                gl.dump(step=t)
                last_log_time, last_log_step = time.time(), t
            if cfg.actual_checkpoints_path is not None and (
                t % cfg.checkpoint_every == 0 or t == cfg.num_train_ops - 1
            ):
                torch.save(agent.state_dict(), os.path.join(cfg.actual_checkpoints_path, f"checkpoint_{t}.pt"))
        if eval_history:
            wandb.run.summary["final/eval_score"] = float(np.mean(eval_history[-cfg.final_evals:]))
    finally:
        wandb.finish()
        for eval_env in eval_envs:
            eval_env.close()


if __name__ == "__main__":
    train()
