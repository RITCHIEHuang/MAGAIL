#!/usr/bin/env python
# Created at 2020/3/27
import os

import numpy as np
import torch
import torch.optim as optim

from models.mlp_actor_sac import Actor
from models.mlp_critic import Value
from rl.FixedMemory import FixedMemory
from rl.sac_alpha_step import sac_alpha_step
from utils.torch_util import device, FLOAT


class SAC_Alpha:
    def __init__(self,
                 env,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_a=3e-4,
                 lr_q=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 target_update_delay=1,
                 seed=1,
                 ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.memory = FixedMemory(memory_size)
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_a = lr_a
        self.lr_q = lr_q
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.target_update_delay = target_update_delay
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        self.action_low, self.action_high = self.env.action_space.low[0], self.env.action_space.high[0]

        self.target_entropy = - np.prod(self.env.action_space.shape)
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Actor(self.num_states, self.num_actions, action_limit=self.action_high).to(device)

        self.q_net_1 = Value(self.num_states + self.num_actions).to(device)
        self.q_net_target_1 = Value(self.num_states + self.num_actions).to(device)
        self.q_net_2 = Value(self.num_states + self.num_actions).to(device)
        self.q_net_target_2 = Value(self.num_states + self.num_actions).to(device)

        # self.alpha init
        self.alpha = torch.exp(torch.zeros(1, device=device)).requires_grad_()

        self.q_net_target_1.load_state_dict(self.q_net_1.state_dict())
        self.q_net_target_2.load_state_dict(self.q_net_2.state_dict())

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_a = optim.Adam([self.alpha], lr=self.lr_a)
        self.optimizer_q_1 = optim.Adam(self.q_net_1.parameters(), lr=self.lr_q)
        self.optimizer_q_2 = optim.Adam(self.q_net_2.parameters(), lr=self.lr_q)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = self.policy_net.get_action_log_prob(state)
        action = action.cpu().numpy()[0]
        return action, None

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            action, _ = self.choose_action(state)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter, step):
        """interact"""

        state = self.env.reset()
        episode_reward = 0

        while True:

            if self.render:
                self.env.render()

            action, _ = self.choose_action(state)

            next_state, reward, done, _ = self.env.step(action)
            mask = 0 if done else 1
            # ('state', 'action', 'reward', 'next_state', 'mask')
            self.memory.push(state, action, reward, next_state, mask)

            episode_reward += reward

            if step >= self.min_update_step and step % self.update_step == 0:
                for k in range(1, self.update_step + 1):
                    batch = self.memory.sample(self.batch_size)  # random sample batch
                    self.update(batch, k)

            if done:
                break

            state = next_state

        self.env.close()

        print(f"Iter: {i_iter}, reward: {episode_reward}")
        # record reward information
        writer.add_scalar("sac_alpha/reward", episode_reward, i_iter)

    def update(self, batch, k_iter):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by SAC Alpha
        sac_alpha_step(self.policy_net, self.q_net_1, self.q_net_2, self.alpha, self.q_net_target_1,
                       self.q_net_target_2,
                       self.optimizer_p, self.optimizer_q_1, self.optimizer_q_2, self.optimizer_a, batch_state,
                       batch_action, batch_reward, batch_next_state, batch_mask, self.gamma, self.polyak,
                       self.target_entropy,
                       k_iter % self.target_update_delay == 0)

    def load(self, model_path):
        print(f"Loading Saved Model from {model_path}")
        self.policy_net, self.q_net_1, self.q_net_2, self.alpha = torch.load(model_path, map_location=device)

    def save(self, save_path):
        """save model"""
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save((self.policy_net, self.q_net_1, self.q_net_2, self.alpha), f"{save_path}/WebEye_sac_alpha.pt")
