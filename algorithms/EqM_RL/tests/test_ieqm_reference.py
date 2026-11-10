"""Numerical regression against the fetched IEqM commit, without dataset downloads.

Run: python -m unittest discover -s tests -p 'test_ieqm_reference.py'
The pinned commit must exist locally (git fetch origin ieqm). No network is used.
"""

import ast
from copy import deepcopy
import io
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "63ec67a09f29b17f64d8b00efe22a42af3f6b895"
sys.path.insert(0, str(ROOT / 'Aware'))
from agents import Otter
from actor_models import Actor
from critic_models import TwinCritic, VCritic
from utils.networks import ConditionalMLP
from buffer import ReplayBuffer
from logger import global_logger
from utils.helper import set_seed, compute_mean_std, modify_reward, wandb_init
from utils import ogbench_data


def archived(path):
    return subprocess.check_output(
        ['git', 'show', f'{REFERENCE}:algorithms/EqM_RL/{path}'], cwd=ROOT, text=True,
    )


def execute_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reference>', 'exec'), namespace)


def reference_namespace():
    trainer = ast.parse(archived('Aware/train_ieqm.py'))
    ns = dict(torch=torch, nn=nn, F=F, np=np)
    execute_nodes([n for n in trainer.body if getattr(n, 'name', None) in
                   {'IFQLMLP', 'IFQLCritic', 'ifql_losses', 'evaluate_d4rl'}], ns)
    ns.update(Module=nn.Module, lin=nn.Linear)
    execute_nodes([n for n in ast.parse(archived('Aware/utils/networks.py')).body
                   if getattr(n, 'name', None) == 'ConditionalMLP'], ns)
    agents = ast.parse(archived('Aware/agents.py'))
    execute_nodes([n for n in agents.body if getattr(n, 'name', None) == 'expand_t_like_x'], ns)
    cls = next(n for n in agents.body if isinstance(n, ast.ClassDef) and n.name == 'Otter')
    names = {'_expand_dims', '_c_linear', '_c_trun', '_sample_gamma_and_noise',
             '_mix_data_and_compute_target', '_fino_mix_data_and_compute_target', '_actor_loss'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    execute_nodes(methods, ns)
    ns['ReferenceEqM'] = type('ReferenceEqM', (), {n.name: ns[n.name] for n in methods})
    train = next(n for n in trainer.body if getattr(n, 'name', None) == 'train')
    ns['update_nodes'] = [n for n in train.body if isinstance(n, ast.FunctionDef)
                          and n.name in {'total_loss', 'update'}]
    actor = next(n for n in ast.parse(archived('Aware/actor_models.py')).body
                 if isinstance(n, ast.ClassDef) and n.name == 'Actor')
    sampler = [n for n in actor.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_implicit', '_descend'}]
    execute_nodes(sampler, ns)
    ns['ReferenceSampler'] = type('ReferenceSampler', (), {n.name: ns[n.name] for n in sampler})
    return ns


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


def config(config_file='config.yaml', **overrides):
    values = yaml.safe_load((ROOT / 'Aware' / config_file).read_text())
    values.update(device='cpu', compile=False, batch_size=8, hidden_dim=16,
                  num_hidden_layers=2, critic_hidden_dim=16, critic_num_hidden_layers=2,
                  buffer_size=32, checkpoints_path=None)
    values['actor']['num_step'] = 3
    values.update(overrides)
    for key, value in values['agent'].items():
        if isinstance(value, str) and value.startswith('${'):
            values['agent'][key] = values[value[2:-1]]
    values['agent'] = Config(values['agent'])
    values['actor'] = Config(values['actor'])
    return Config(values)


def reference_config(cfg):
    values = dict(cfg)
    values.pop('agent')
    values.pop('actor')
    values.update(critic_n_hidden=cfg.critic_num_hidden_layers,
                  eval_candidates=cfg.num_actions_inference,
                  flag_type=cfg.agent.flag_type, flag_threshold=cfg.agent.flag_threshold,
                  scale_gradient=cfg.agent.scale_gradient,
                  num_step=cfg.actor.num_step, step_size=cfg.actor.step_size, moment=cfg.actor.moment)
    return SimpleNamespace(**values)


def create_agent(cfg, mean, std):
    value = VCritic(3, cfg.critic_hidden_dim, cfg.critic_num_hidden_layers)
    critic = TwinCritic(3, 2, cfg.critic_hidden_dim, cfg.critic_num_hidden_layers)
    model = ConditionalMLP(5, 2, cfg.hidden_dim, cfg.num_hidden_layers)
    optimizer = torch.optim.Adam([*value.parameters(), *critic.parameters(), *model.parameters()],
                                 lr=cfg.learning_rate, fused=False)
    mean, std = torch.tensor(mean), torch.tensor(std)
    actor = Actor(model=model, q_nets=critic, v_net=value, action_dim=2,
                  min_action=-1., max_action=1., state_mean=mean, state_std=std, **cfg.actor)
    agent = Otter(model=model, model_optimizer=optimizer, actor=actor, q_nets=critic, v_net=value,
                  state_mean=mean, state_std=std, batch_size=cfg.batch_size, **cfg.agent)
    if cfg.compile:
        agent.compile_losses(mode=cfg.compile_mode)
    return agent


def reference_state(local):
    """Translate only checkpoint names; every tensor must still agree exactly."""
    return {
        'model': local['model'],
        'value': {key.replace('model.model.', 'net.'): value for key, value in local['v_net'].items()},
        'critic': {key.replace('.model.', '.net.'): value for key, value in local['q_nets'].items()},
        'target': {key.replace('.model.', '.net.'): value for key, value in local['target_q_nets'].items()},
    }


def reference_metrics(local, prefixed=False):
    renamed = {}
    for key, value in local.items():
        key = key if prefixed else 'train/' + key
        key = {'train/q_critic_loss': 'train/q_loss', 'train/v_critic_loss': 'train/v_loss'}.get(key, key)
        renamed[key] = value
    return renamed


def dataset():
    rng = np.random.default_rng(7)
    return dict(observations=rng.normal(size=(24, 3)).astype(np.float32),
                next_observations=rng.normal(size=(24, 3)).astype(np.float32),
                actions=rng.uniform(-2, 2, (24, 2)).astype(np.float32),
                rewards=rng.normal(size=24).astype(np.float32),
                terminals=np.array([0, 0, 1] * 8, dtype=np.float32))


def setup_pair(cfg):
    data = dataset()
    mean, std = data['observations'].mean(0), data['observations'].std(0) + 1e-3
    torch.manual_seed(cfg.seed)
    agent = create_agent(cfg, mean, std)
    cfg = reference_config(cfg)
    local_rng_state = torch.get_rng_state()
    ns = reference_namespace()
    torch.manual_seed(cfg.seed)
    value = ns['IFQLMLP'](3, cfg.critic_hidden_dim, cfg.critic_n_hidden)
    critic = ns['IFQLCritic'](3, 2, cfg.critic_hidden_dim, cfg.critic_n_hidden)
    target = deepcopy(critic).requires_grad_(False)
    model = ns['ConditionalMLP'](5, 2, cfg.hidden_dim, cfg.num_hidden_layers)
    opt = torch.optim.Adam([*value.parameters(), *critic.parameters(), *model.parameters()],
                           lr=cfg.learning_rate, fused=False)
    eqm = ns['ReferenceEqM']()
    eqm._model, eqm.flag_type = model, cfg.flag_type
    eqm.flag_threshold, eqm.scale_gradient, eqm.fino = cfg.flag_threshold, cfg.scale_gradient, False
    ns.update(value=value, critic=critic, target=target, model=model, opt=opt, eqm=eqm, cfg=cfg,
              mean_t=torch.tensor(mean), std_t=torch.tensor(std),
              ONES=torch.ones(cfg.batch_size, 1), t_params=list(target.parameters()),
              c_params=list(critic.parameters()), S=torch.tensor(data['observations']),
              S2=torch.tensor(data['next_observations']), R=torch.tensor(data['rewards']),
              D=torch.tensor(data['terminals']), A=torch.tensor(np.clip(data['actions'],
                  -1 + cfg.action_clip_eps, 1 - cfg.action_clip_eps)))
    execute_nodes(ns['update_nodes'], ns)
    ns['loss_fn'] = ns['total_loss']
    assert torch.equal(torch.get_rng_state(), local_rng_state)
    buffer = ReplayBuffer(3, 2, 32, action_clip_eps=cfg.action_clip_eps)
    buffer.load_d4rl_dataset(data)
    return agent, ns, buffer


class ToyEnv:
    """Deterministic episodes with different lengths and both forms of episode ending."""

    def __init__(self, modern=False):
        self.modern = modern
        self.closed = False
        self.action_space = SimpleNamespace(low=np.full(2, -1.), high=np.ones(2), seed=lambda seed: None)
        self.spec = SimpleNamespace(max_episode_steps=4)

    def seed(self, seed):
        self.episode_seed = seed

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.state = np.array([self.episode_seed / 100, 0., 1.], dtype=np.float64)
        return (self.state.copy(), {}) if self.modern else self.state.copy()

    def step(self, action):
        self.t += 1
        self.state += np.array([action[0], .8, action[1]])
        reward = float(self.state[0] - np.square(action).sum())
        done = self.t >= 2 + self.episode_seed % 3
        if self.modern:
            terminal = done and self.episode_seed % 2 == 0
            return self.state.copy(), reward, terminal, done and not terminal, {}
        return self.state.copy(), reward, done, {}

    def get_normalized_score(self, score):
        return score / 10

    def close(self):
        self.closed = True


class ToyOGBenchEnv(ToyEnv):
    def __init__(self):
        super().__init__(modern=True)
        self.unwrapped = SimpleNamespace(_reward_task_id=1, num_tasks=5)

    def reset(self, seed=None):
        super().reset(seed)
        self.state += np.random.normal(size=3)
        return self.state.copy(), {}

    def step(self, action):
        observation, _, terminated, truncated, _ = super().step(action)
        # An intermediate success must not count when the final step fails.
        success = float(terminated or not (terminated or truncated))
        return observation, -1.0 + float(terminated), terminated, truncated, {'success': success}


def reference_ogbench():
    module = ModuleType('reference_ogbench_data')
    exec(compile(archived('Aware/ogbench_data.py'), '<reference_ogbench>', 'exec'), module.__dict__)
    return module


def run_training(variant, ogbench=False):
    """Run the actual training loop; stub only datasets, environments, and external logging/config APIs."""
    modern = variant == 'Aware_Minari'
    envs, logs = [], []
    data = dataset()
    mean, std = data['observations'].mean(0), data['observations'].std(0) + 1e-3

    def make_env(*args):
        env = ToyOGBenchEnv() if ogbench else ToyEnv(modern)
        envs.append(env)
        return env

    gym, d4rl, minari = ModuleType('gym'), ModuleType('d4rl'), ModuleType('minari_utils')
    gym.make = make_env
    d4rl.qlearning_dataset = lambda env: data
    md = {**data, 'terminations': data['terminals'], 'truncations': np.ones(24)}
    minari.prepare_minari_data = lambda name: (make_env(), md, SimpleNamespace(recover_environment=make_env))
    minari.get_ref_scores = lambda dataset: (0., 10.)
    minari.minari_normalized_score = lambda score, refs: score / 10 * 100
    og = ModuleType('ogbench')
    og_data = {**data, 'masks': 1.0 - data['terminals'],
               'terminals': np.array([0, 1, 0, 0] * 6, dtype=np.float32)}

    def make_ogbench(name, env_only=False, **kwargs):
        env = make_env()
        return env if env_only else (env, og_data, {})

    og.make_env_and_datasets = make_ogbench
    wb = SimpleNamespace(run=SimpleNamespace(summary={}), init=lambda **kw: None,
                         log=lambda values, step: logs.append((step, values.copy())), finish=lambda: None)

    def save_config(config, f):
        contents = yaml.safe_dump(plain(config))
        if hasattr(f, 'write'):
            f.write(contents)
        else:
            Path(f).write_text(contents)

    omega = SimpleNamespace(to_container=lambda cfg, resolve: plain(cfg),
                            set_struct=lambda cfg, flag: None, save=save_config)
    if variant == 'reference':
        ns = reference_namespace()

        def make_eqm(**kw):
            obj = ns['ReferenceEqM']()
            obj._model = kw['model']
            obj.flag_type, obj.flag_threshold = kw['flag_type'], kw['flag_threshold']
            obj.scale_gradient, obj.fino = kw['scale_gradient'], False
            return obj

        def make_sampler(**kw):
            obj = ns['ReferenceSampler']()
            obj.__dict__.update(kw)
            obj._compiled = False
            obj._descend_fn = obj._descend
            return obj

        ns.update(Otter=make_eqm, Actor=make_sampler, deepcopy=deepcopy, asdict=vars, TrainConfig=SimpleNamespace,
                  ogbench_data=reference_ogbench(),
                  load_d4rl=lambda name: (make_env(), data, mean, std, np.arange(24) // 3))
        source = archived('Aware/train_ieqm.py')
    else:
        ns = dict(Otter=Otter, ReplayBuffer=ReplayBuffer, gl=global_logger,
                  OmegaConf=omega, DictConfig=Config, Actor=Actor,
                  TwinCritic=TwinCritic, VCritic=VCritic, ConditionalMLP=ConditionalMLP,
                  compute_mean_std=compute_mean_std, modify_reward=modify_reward, wandb_init=wandb_init,
                  ogbench_data=ogbench_data,
                  trange=lambda start, stop, **kw: range(start, stop))
        source = (ROOT / variant / 'main.py').read_text()
    ns.update(torch=torch, np=np, wandb=wb, os=os, time=time, uuid=uuid, json=json, yaml=yaml, set_seed=set_seed)
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {'train', 'evaluate_envs', 'evaluate_actor'}]
    for node in nodes:
        node.decorator_list = []
    execute_nodes(nodes, ns)
    stdout = io.StringIO()
    with tempfile.TemporaryDirectory() as directory:
        config_file = 'configs/config_og.yaml' if ogbench else 'configs/config_d4rl.yaml'
        cfg = config(config_file=config_file, num_train_ops=10, eval_frequency=4, log_frequency=3, stats_every=2,
                     checkpoint_every=4, n_test_episodes=3, final_evals=2,
                     frozen_bc_prefix='', seed=2, checkpoints_path=directory,
                     env_name='mujoco/halfcheetah/simple-v0' if modern else 'halfcheetah-medium-v2')
        if ogbench:
            cfg.env_name = 'antmaze-large-navigate-singletask-v0'
            cfg.ogbench_dir, cfg.eval_tasks = '/unused/ogbench_data', 'own'
        with patch.dict(sys.modules, gym=gym, d4rl=d4rl, minari_utils=minari, ogbench=og), \
             patch('logger.wandb', wb), patch('utils.helper.wandb', wb), contextlib.redirect_stdout(stdout):
            ns['train'](reference_config(cfg) if variant == 'reference' else cfg)
        checkpoints = {p.name: torch.load(p, weights_only=True) for p in Path(directory).rglob('checkpoint_*.pt')}
        config_paths = list(Path(directory).rglob('config.yaml'))
        assert len(config_paths) == 1 and config_paths[0].parent.parent == Path(directory)
    assert len(envs) == cfg.n_test_episodes + int(ogbench)
    if variant != 'reference' or not ogbench:
        assert all(e.closed for e in envs)
    return logs, wb.run.summary, checkpoints, stdout.getvalue()


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_same(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_implementation_and_config(self):
        def executable_tree(path):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) is not None:
                        node.body.pop(0)
            return ast.dump(tree)

        for name in ['agents.py', 'actor_models.py', 'critic_models.py', 'buffer.py',
                     'logger.py', 'utils/networks.py']:
            self.assertEqual(executable_tree(ROOT / 'Aware' / name),
                             executable_tree(ROOT / 'Aware_Minari' / name), name)
        configs = [yaml.safe_load((ROOT / f / 'config.yaml').read_text()) for f in ['Aware', 'Aware_Minari']]
        for cfg in configs:
            cfg.pop('env_name')
            cfg.pop('group')
        self.assertEqual(*configs)
        cfg = Config(configs[0])
        cfg.agent, cfg.actor = Config(cfg.agent), Config(cfg.actor)
        defaults = vars(reference_config(cfg))
        expected = yaml.safe_load(archived('sweeps/exp23_d4rl_ieqm_halfcheetah.yaml'))['fixed']
        for key in ['gamma', 'num_train_ops', 'eval_frequency', 'n_test_episodes', 'final_evals',
                    'eval_candidates', 'num_step', 'step_size', 'moment', 'hidden_dim', 'num_hidden_layers',
                    'critic_hidden_dim', 'critic_n_hidden', 'batch_size', 'learning_rate', 'tau',
                    'scale_gradient', 'flag_threshold', 'checkpoint_every', 'compile']:
            self.assertEqual(defaults[key], expected[key], key)

    def test_initialization_losses_gradients_updates_and_rng(self):
        for expectile in [.8, .9]:
            cfg = config(expectile=expectile)
            agent, ref, buffer = setup_pair(cfg)
            ref_models = dict(model=ref['model'], value=ref['value'], critic=ref['critic'], target=ref['target'])
            for name, model in ref_models.items():
                for key, tensor in model.state_dict().items():
                    self.assert_same(reference_state(agent.state_dict())[name][key], tensor)
            local_gen = torch.Generator().manual_seed(cfg.seed + 1000)
            ref_gen = torch.Generator().manual_seed(cfg.seed + 1000)
            for step in range(6):
                batch = buffer.sample(cfg.batch_size, local_gen)
                indices = torch.randint(0, len(ref['S']), (cfg.batch_size,), generator=ref_gen)
                for local, expected in zip(batch, [ref['S'][indices], ref['A'][indices], ref['R'][indices, None],
                        ref['S2'][indices], torch.zeros(cfg.batch_size, 1), ref['D'][indices, None]]):
                    self.assert_same(local, expected)
                before = torch.get_rng_state()
                actual = reference_metrics(agent.update(batch))
                after = torch.get_rng_state()
                torch.set_rng_state(before)
                expected = ref['update'](indices, True)
                self.assert_same(torch.get_rng_state(), after)
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    self.assert_same(actual[key], expected[key])
                    self.assertFalse(actual[key].requires_grad)
                for name, model in ref_models.items():
                    for key, tensor in model.state_dict().items():
                        self.assert_same(reference_state(agent.state_dict())[name][key], tensor)
                for local, original in [(agent._model, ref['model']), (agent._v_net, ref['value']),
                                         (agent._q_nets, ref['critic'])]:
                    for lp, rp in zip(local.parameters(), original.parameters()):
                        self.assert_same(lp.grad, rp.grad)
                self.assertTrue(all(p.grad is None and not p.requires_grad for p in agent._target_q_nets.parameters()))

    def test_batched_selection_matches_reference_and_preserves_training_rng(self):
        cfg = config()
        agent, ref, _ = setup_pair(cfg)
        sampler = ref['ReferenceSampler']()
        sampler.model, sampler._compiled = ref['model'], False
        sampler.opt_type, sampler.num_step = 'nag', cfg.actor.num_step
        sampler.step_size, sampler.moment = cfg.actor.step_size, cfg.actor.moment
        sampler._descend_fn = sampler._descend
        ours_rng = torch.Generator().manual_seed(cfg.seed + 2000)
        ref_rng = torch.Generator().manual_seed(cfg.seed + 2000)
        before = torch.get_rng_state()
        for size in [4, 2, 1]:
            s = ref['S'][:size]
            actual, info = agent._actor.sample_rejection(s.numpy(), cfg.num_actions_inference, ours_rng, True)
            with torch.no_grad():
                x = torch.randn(size * cfg.num_actions_inference, 2, generator=ref_rng)
                cand = sampler._implicit(x, ((s - ref['mean_t']) / ref['std_t']).repeat_interleave(cfg.num_actions_inference, 0))
                cand = cand.clamp(-1, 1).view(size, cfg.num_actions_inference, 2)
                q = ref['critic'](s[:, None].expand(-1, cfg.num_actions_inference, -1), cand)
                qb, best = q.max(1)
                expected = cand[torch.arange(size), best].numpy()
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(info['q_start'], qb.mean().item())
            self.assertEqual(info['gain'], (qb - q.mean(1)).sum().item())
        self.assert_same(torch.get_rng_state(), before)
        self.assert_same(ours_rng.get_state(), ref_rng.get_state())

    def test_checkpoints_and_minari_masks(self):
        cfg = config()
        agent, ref, buffer = setup_pair(cfg)
        agent.update(buffer.sample(cfg.batch_size, torch.Generator().manual_seed(1001)))
        checkpoint = io.BytesIO()
        torch.save(agent.state_dict(), checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        self.assertEqual(set(saved), {'model', 'q_nets', 'target_q_nets', 'v_net'})
        restored = create_agent(cfg, agent._state_mean.numpy(), agent._state_std.numpy())
        restored.load_state_dict(saved)
        self.assertEqual(restored.state_dict().keys(), saved.keys())
        saved_reference = reference_state(saved)
        for key in ['model', 'value', 'critic', 'target']:
            ref[key].load_state_dict(saved_reference[key])
            for name, tensor in ref[key].state_dict().items():
                self.assert_same(tensor, reference_state(restored.state_dict())[key][name])
        data = dataset()
        data['terminations'] = data.pop('terminals')
        data['truncations'] = np.ones(24, dtype=np.float32)
        minari_buffer = ReplayBuffer(3, 2, 32)
        minari_buffer.load_minari_dataset(data)
        np.testing.assert_array_equal(minari_buffer._states[:24], data['observations'])
        np.testing.assert_array_equal(minari_buffer._terminals[:24, 0], data['terminations'])
        np.testing.assert_array_equal(minari_buffer._truncations[:24, 0], data['truncations'])
        self.assertEqual(minari_buffer._rewards.shape, (32, 1))
        self.assertLessEqual(minari_buffer._actions[:24].max().item(), np.float32(1 - 1e-5))

    def test_compiled_loss_wiring(self):
        cfg = config(compile=True)
        real_compile = torch.compile
        calls = []
        def tracing(fn, mode):
            calls.append(mode)
            return real_compile(fn, backend='eager', fullgraph=True)
        with patch('torch.compile', side_effect=tracing):
            agent, ref, buffer = setup_pair(cfg)
            batch = buffer.sample(cfg.batch_size, torch.Generator().manual_seed(3))
            before = torch.get_rng_state()
            actual = reference_metrics(agent.update(batch))
            torch.set_rng_state(before)
            # Compare the traced combined loss using the same data, not codegen performance.
            expected = ref['total_loss'](batch[0], batch[1], batch[2].squeeze(-1), batch[5].squeeze(-1), batch[3])
            self.assert_same(actual['train/v_loss'], expected[1])
            self.assert_same(actual['train/q_loss'], expected[2])
            self.assert_same(actual['train/actor_loss'], expected[3])
            self.assertEqual(calls, ['reduce-overhead'])
            self.assertFalse(agent._actor._compiled)

    def test_training_loops_match_reference(self):
        expected_logs, expected_summary, expected_checkpoints, _ = run_training('reference')
        self.assertEqual([t for t, _ in expected_logs], [3, 4, 6, 8, 9])
        self.assertEqual(set(expected_checkpoints), {'checkpoint_4.pt', 'checkpoint_8.pt', 'checkpoint_9.pt'})
        for variant in ['Aware', 'Aware_Minari']:
            actual_logs, summary, checkpoints, output = run_training(variant)
            self.assertEqual(summary, expected_summary)
            self.assertEqual([t for t, _ in actual_logs], [t for t, _ in expected_logs])
            for (_, actual), (_, expected) in zip(actual_logs, expected_logs):
                actual = reference_metrics(actual, prefixed=True)
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    if key not in {'train/ms_per_step', 'eval/seconds'}:
                        self.assertEqual(actual[key], expected[key], (variant, key))
            self.assertEqual(checkpoints.keys(), expected_checkpoints.keys())
            for filename, state in checkpoints.items():
                state = reference_state(state)
                for model in state:
                    for key, tensor in state[model].items():
                        self.assert_same(tensor, expected_checkpoints[filename][model][key])
            self.assertEqual([t for t, values in actual_logs if 'eval/score' in values], [4, 8, 9])
            last_two = [values['eval/normalized_score'] for _, values in actual_logs
                        if 'eval/normalized_score' in values][-2:]
            self.assertEqual(summary['final/eval_score'], float(np.mean(last_two)))

    def test_ogbench_loading_and_evaluation_match_reference(self):
        reference = reference_ogbench()
        source = dataset()
        source['masks'] = 1.0 - source['terminals']
        source['terminals'] = np.array([0, 1, 0, 0] * 6, dtype=np.float32)
        module = ModuleType('ogbench')
        module.make_env_and_datasets = lambda *args, **kwargs: (ToyOGBenchEnv(), source, {})
        with patch.dict(sys.modules, ogbench=module):
            env, actual = ogbench_data.load_ogbench('antmaze-large-navigate-singletask-v0', '/unused')
            _, expected, *_ = reference.load_ogbench('antmaze-large-navigate-singletask-v0', '/unused')
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        self.assertFalse(np.array_equal(actual['terminals'], source['terminals']))
        for spec in ['own', 'all', '1,3']:
            self.assertEqual(ogbench_data.parse_tasks(spec, env), reference.parse_tasks(spec, env))

        def evaluate(helper, original):
            states = []
            envs = [(task, ToyOGBenchEnv()) for task in [1, 1, 3, 3]]

            def act(observations):
                states.append(observations.copy())
                return observations[:, :2] * 0.2

            np.random.seed(42)
            before = np.random.get_state()
            result = helper.evaluate(envs, act, 0., 1., 69, -1., 1.) if original else \
                helper.evaluate(envs, act, 69, -1., 1.)
            after = np.random.get_state()
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            return result, states, [env.episode_seed for _, env in envs]

        actual, actual_states, seeds = evaluate(ogbench_data, False)
        expected, expected_states, _ = evaluate(reference, True)
        self.assertEqual(seeds, [1069, 1070, 3069, 3070])
        self.assertEqual(actual, expected)
        self.assertEqual(actual['normalized_score'], 50.)
        for a, b in zip(actual_states, expected_states):
            np.testing.assert_array_equal(a, b)

    def test_ogbench_training_loop_matches_reference(self):
        expected_logs, expected_summary, expected_checkpoints, _ = run_training('reference', ogbench=True)
        actual_logs, summary, checkpoints, _ = run_training('Aware', ogbench=True)
        self.assertEqual(summary, expected_summary)
        self.assertEqual([step for step, _ in actual_logs], [step for step, _ in expected_logs])
        for (_, actual), (_, expected) in zip(actual_logs, expected_logs):
            actual = reference_metrics(actual, prefixed=True)
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                if key not in {'train/ms_per_step', 'eval/seconds'}:
                    self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(checkpoints.keys(), expected_checkpoints.keys())
        for filename, state in checkpoints.items():
            for model, values in reference_state(state).items():
                for key, tensor in values.items():
                    self.assert_same(tensor, expected_checkpoints[filename][model][key])


if __name__ == '__main__':
    unittest.main()
