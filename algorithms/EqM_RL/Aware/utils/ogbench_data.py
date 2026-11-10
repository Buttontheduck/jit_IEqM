"""OGBench single-task loading and evaluation, matching the fetched IEqM runs."""

import os

import numpy as np


def is_ogbench(env_name: str) -> bool:
    return "singletask" in env_name.split("-")


def task_env_name(env_name: str, task: int) -> str:
    parts = env_name.split("-")
    position = parts.index("singletask")
    return "-".join(parts[:position + 1] + [f"task{task}", parts[-1]])


def parse_tasks(spec: str, env) -> list:
    if spec == "own":
        return [int(env.unwrapped._reward_task_id)]
    if spec == "all":
        return list(range(1, env.unwrapped.num_tasks + 1))
    tasks = [int(task) for task in spec.split(",") if task.strip()]
    if not tasks or any(task < 1 or task > env.unwrapped.num_tasks for task in tasks):
        raise ValueError(f"Invalid eval_tasks: {spec!r}")
    return tasks


def load_ogbench(env_name: str, dataset_dir: str):
    import ogbench

    env, train, _ = ogbench.make_env_and_datasets(
        env_name, dataset_dir=os.path.expanduser(dataset_dir), compact_dataset=False,
    )
    dataset = {
        "observations": train["observations"],
        "actions": train["actions"],
        "rewards": train["rewards"],
        "next_observations": train["next_observations"],
        # OGBench terminals mark trajectory boundaries; masks mark task completion.
        "terminals": (1.0 - train["masks"]).astype(np.float32),
    }
    return env, dataset


def make_eval_envs(env_name: str, tasks: list, n_per_task: int) -> list:
    import ogbench

    envs = []
    for task in tasks:
        name = task_env_name(env_name, task)
        for _ in range(n_per_task):
            env = ogbench.make_env_and_datasets(name, env_only=True)
            envs.append((task, env))
    return envs


def evaluate(envs: list, act, seed: int, low: float, high: float) -> dict:
    """Pass raw states to act; score success at each episode's final step."""
    num_episodes = len(envs)
    tasks = np.array([task for task, _ in envs])
    rewards = np.zeros(num_episodes)
    lengths = np.zeros(num_episodes, dtype=int)
    done = np.zeros(num_episodes, dtype=bool)
    success = np.zeros(num_episodes)
    numpy_state = np.random.get_state()

    try:
        observations = []
        task_episode = {}
        for task, env in envs:
            episode = task_episode.get(task, 0)
            task_episode[task] = episode + 1
            episode_seed = seed + 1000 * task + episode
            # Maze start noise also uses NumPy's global generator.
            np.random.seed(episode_seed % 2 ** 32)
            observation, _ = env.reset(seed=episode_seed)
            observations.append(observation)
        observations = np.stack(observations).astype(np.float64)

        while not done.all():
            active = np.flatnonzero(~done)
            actions = act(observations[active].astype(np.float32))
            actions = np.clip(actions, low, high)
            for index, episode in enumerate(active):
                env = envs[episode][1]
                observation, reward, terminated, truncated, info = env.step(actions[index])
                observations[episode] = observation
                rewards[episode] += reward
                lengths[episode] += 1
                if terminated or truncated:
                    done[episode] = True
                    success[episode] = float(info["success"])
    finally:
        np.random.set_state(numpy_state)

    result = {}
    task_scores = []
    for task in sorted(set(tasks)):
        selected = tasks == task
        score = 100.0 * success[selected].mean()
        task_scores.append(score)
        result[f"success_task{task}"] = score
        result[f"return_task{task}"] = rewards[selected].mean()
        result[f"length_task{task}"] = lengths[selected].mean()
    result.update(
        normalized_score=float(np.mean(task_scores)),
        score=rewards.mean(),
        score_std=rewards.std(),
        score_min=rewards.min(),
        episode_length=lengths.mean(),
    )
    return result
