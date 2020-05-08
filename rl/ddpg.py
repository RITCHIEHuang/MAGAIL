#!/usr/bin/env python
# Created at 2020/5/6

import os

import numpy as np
import torch
import torch.optim as optim

from models.mlp_actor_deterministic import Actor
from models.mlp_critic import Value
from rl.FixedMemory import FixedMemory
from rl.ddpg_step import ddpg_step
from utils.torch_util import FLOAT, device


class DDPG:
    def __init__(self,
                 env=None,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_v=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 explore_size=10000,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 action_noise=0.1,
                 seed=1,
                 ):
        self.env = env
        self.render = render
        self.gamma = gamma
        self.polyak = polyak
        self.memory = FixedMemory(memory_size)
        self.explore_size = explore_size
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.action_noise = action_noise
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        self.action_low, self.action_high = self.env.action_space.low[0], self.env.action_space.high[0]
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Actor(self.num_states, self.num_actions, self.action_high).to(device)
        self.policy_net_target = Actor(self.num_states, self.num_actions, self.action_high).to(device)

        self.value_net = Value(self.num_states + self.num_actions).to(device)
        self.value_net_target = Value(self.num_states + self.num_actions).to(device)

        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.value_net_target.load_state_dict(self.value_net.state_dict())

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=self.lr_v)

    def choose_action(self, state, noise_scale):
        """select action"""
        self.policy_net.eval()
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.policy_net(state)
        self.policy_net.train()
        action = action.cpu().numpy()[0]
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action += noise
        action = np.clip(action, -self.action_high, self.action_high)
        return action

    def eval(self, i_iter, render=False):
        """evaluate model"""
        self.policy_net.eval()
        self.value_net.eval()

        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            action = self.choose_action(state, 0)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter, step):
        """interact"""
        self.policy_net.train()
        self.value_net.train()

        state = self.env.reset()
        episode_reward = 0

        while True:
            if self.render:
                self.env.render()

            action = self.choose_action(state, self.action_noise)

            next_state, reward, done, _ = self.env.step(action)
            mask = 0 if done else 1
            # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
            self.memory.push(state, action, reward, next_state, mask)

            episode_reward += reward

            if step >= self.min_update_step and step % self.update_step == 0:
                for _ in range(self.update_step):
                    batch = self.memory.sample(self.batch_size)  # random sample batch
                    self.update(batch)

            if done:
                break

            state = next_state

        self.env.close()

        print(f"Iter: {i_iter}, reward: {episode_reward}")

        # record reward information
        writer.add_scalar("ddpg/reward", episode_reward, i_iter)

    def update(self, batch):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by DDPG
        ddpg_step(self.policy_net, self.policy_net_target, self.value_net, self.value_net_target, self.optimizer_p,
                  self.optimizer_v, batch_state, batch_action, batch_reward, batch_next_state, batch_mask,
                  self.gamma, self.polyak)

    def load(self, model_path):
        print(f"Loading Saved Model from {model_path}")
        self.policy_net, self.value_net = torch.load(model_path, map_location=device)

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        """save model"""
        torch.save((self.policy_net, self.value_net), f"{save_path}/WebEye_ddpg.pt")
