#!/usr/bin/env python
# Created at 2020/3/10
import math
import multiprocessing
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import time
import numpy as np

from algos.GAE import estimate_advantages
from algos.ppo_step import ppo_step
from data.ExpertDataSet import ExpertDataSet
from algos.JointPolicy import JointPolicy
from models.mlp_critic import Value
from models.mlp_discriminator import Discriminator

class MAGAIL:
    def __init__(self, expert_data_path, config, log_dir):
        self.expert_data_path = expert_data_path
        self.config = config
        self.exp_name = f"Magail_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.exp_name}")

        """seeding"""
        seed = self.config["general"]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._load_expert_data()
        self._init_model()

    def _init_model(self):
        self.V = Value(num_states=self.config["value"]["num_states"],
                       num_hiddens=self.config["value"]["num_hiddens"],
                       drop_rate=self.config["value"]["drop_rate"])
        self.P = JointPolicy(initial_state=self.expert_data_set.state,
                              config=self.config["policy"])
        self.D = Discriminator(num_states=self.config["discriminator"]["num_states"],
                               num_actions=self.config["discriminator"]["num_actions"],
                               num_hiddens=self.config["discriminator"]["num_hiddens"],
                               drop_rate=self.config["discriminator"]["drop_rate"])

        self.optimizer_policy = optim.Adam(self.P.parameters(), lr=self.config["policy"]["learning_rate"])
        self.optimizer_value = optim.Adam(self.V.parameters(), lr=self.config["value"]["learning_rate"])
        self.optimzer_discriminator = optim.Adam(self.D.parameters(), lr=self.config["discriminator"]["learning_rate"])

        self.discriminator_func = nn.BCELoss()

    def _load_expert_data(self):
        num_expert_states = self.config["policy"]["num_states"]
        num_expert_actions = self.config["policy"]["num_actions"]
        self.expert_data_set = ExpertDataSet(data_set_path=self.expert_data_path, num_states=num_expert_states,
                                             num_actions=num_expert_actions)
        expert_batch_size = self.config["general"]["expert_batch_size"]
        self.expert_data_loader = DataLoader(dataset=self.expert_data_set, batch_size=expert_batch_size,
                                             shuffle=True, num_workers=multiprocessing.cpu_count() // 2)

    def train(self, epoch):
        self.P.train()
        self.D.train()
        self.V.train()

        # collect generated batch
        gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
        # batch: ('state', 'action', 'next_state', 'log_prob', 'mask')
        gen_batch_state = torch.stack(gen_batch.state)
        gen_batch_action = torch.stack(gen_batch.action)
        gen_batch_next_state = torch.stack(gen_batch.next_state)
        gen_batch_old_log_prob = torch.stack(gen_batch.log_prob)
        gen_batch_mask = torch.stack(gen_batch.mask)

        # grad_collect_func = lambda d: torch.cat([grad.view(-1) for grad in torch.autograd.grad(d, self.D.parameters(), retain_graph=True)]).unsqueeze(0)
        ####################################################
        # update discriminator
        ####################################################
        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            # gaussian noise for Discriminator input is not necessary, it's a trick for tuning.
            # noise_gen_state = torch.normal(0, self.config["general"]["noise_std"], size=gen_batch_state.size())
            # noise_gen_action = torch.normal(0, self.config["general"]["noise_std"], size=gen_batch_action.size())
            # noise_expert_state = torch.normal(0, self.config["general"]["noise_std"], size=expert_batch_state.size())
            # noise_expert_action = torch.normal(0, self.config["general"]["noise_std"], size=expert_batch_action.size())
            gen_r = self.D(gen_batch_state, gen_batch_action)
            expert_r = self.D(expert_batch_state, expert_batch_action)

            e_loss = self.discriminator_func(expert_r, torch.ones_like(expert_r))
            g_loss = self.discriminator_func(gen_r, torch.zeros_like(gen_r))
            d_loss = e_loss + g_loss

            # """ WGAN with Gradient Penalty"""
            # d_loss = gen_r.mean() - expert_r.mean()
            # differences_batch_state = gen_batch_state[:expert_batch_state.size(0)] - expert_batch_state
            # differences_batch_action = gen_batch_action[:expert_batch_action.size(0)] - expert_batch_action
            # alpha = torch.rand(expert_batch_state.size(0), 1)
            # interpolates_batch_state = gen_batch_state[:expert_batch_state.size(0)] + (alpha * differences_batch_state)
            # interpolates_batch_action = gen_batch_action[:expert_batch_action.size(0)] + (alpha * differences_batch_action)
            # gradients = torch.cat([x for x in map(grad_collect_func, self.D(interpolates_batch_state, interpolates_batch_action))])
            # slopes = torch.norm(gradients, p=2, dim=-1)
            # gradient_penalty = torch.mean((slopes - 1.) ** 2)
            # d_loss += 10 * gradient_penalty

            self.optimzer_discriminator.zero_grad()
            d_loss.backward()
            self.optimzer_discriminator.step()

        self.writer.add_scalar('d_loss', d_loss.item(), epoch)
        self.writer.add_scalar('expert_r', expert_r.mean().item(), epoch)
        self.writer.add_scalar('gen_r', gen_r.mean().item(), epoch)

        with torch.no_grad():
            # noise_gen_state = torch.normal(0, self.config["general"]["noise_std"], size=gen_batch_state.size())
            # noise_gen_action = torch.normal(0, self.config["general"]["noise_std"], size=gen_batch_action.size())
            gen_batch_value = self.V(gen_batch_state)
            gen_batch_reward = self.D(gen_batch_state, gen_batch_action)

        gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                    gen_batch_value, self.config["gae"]["gamma"],
                                                                    self.config["gae"]["tau"])

        ####################################################
        # update policy by ppo [mini_batch]
        ####################################################
        ppo_optim_epochs = self.config["ppo"]["ppo_optim_epochs"]
        ppo_mini_batch_size = self.config["ppo"]["ppo_mini_batch_size"]
        gen_batch_size = gen_batch_state.shape[0]
        optim_iter_num = int(math.ceil(gen_batch_size / ppo_mini_batch_size))

        for _ in range(ppo_optim_epochs):
            perm = torch.randperm(gen_batch_size)

            for i in range(optim_iter_num):
                ind = perm[slice(i * ppo_mini_batch_size,
                                 min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                mini_batch_state, mini_batch_action, mini_batch_next_state, mini_batch_advantage, mini_batch_return, \
                mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], \
                                          gen_batch_next_state[ind], \
                                          gen_batch_advantage[ind], \
                                          gen_batch_return[ind], gen_batch_old_log_prob[ind]

                v_loss, p_loss = ppo_step(self.P, self.V, self.optimizer_policy, self.optimizer_value,
                                          states=mini_batch_state,
                                          actions=mini_batch_action,
                                          next_states=mini_batch_next_state,
                                          returns=mini_batch_return,
                                          old_log_probs=mini_batch_old_log_prob,
                                          advantages=mini_batch_advantage,
                                          ppo_clip_ratio=self.config["ppo"]["clip_ratio"],
                                          value_l2_reg=self.config["value"]["l2_reg"])

                self.writer.add_scalar('p_loss', p_loss, epoch)
                self.writer.add_scalar('v_loss', v_loss, epoch)

        print(f" Training episode:{epoch} ".center(80, "#"))
        print('gen_r:', gen_r.mean().item())
        print('expert_r:', expert_r.mean().item())
        print('d_loss', d_loss.item())

    def eval(self, epoch):
        self.P.eval()
        self.D.eval()
        self.V.eval()

        gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
        gen_batch_state = torch.stack(gen_batch.state)
        gen_batch_action = torch.stack(gen_batch.action)

        for expert_batch_state, expert_batch_action in self.expert_data_loader:
            gen_r = self.D(gen_batch_state, gen_batch_action)
            expert_r = self.D(expert_batch_state, expert_batch_action)

            print(f" Evaluating episode:{epoch} ".center(80, "-"))
            print('gen_r:', gen_r.mean().item())
            print('expert_r:', expert_r.mean().item())

            break

    def save_model(self, save_path):
        # dump model from pkl file
        pickle.dump((self.D, self.P, self.V), open(f"{save_path}/{self.exp_name}.pkl", 'wb'))

    def load_model(self, model_path):
        # load entire model
        self.D, self.P, self.V = pickle.load(open(model_path, 'wb'))
