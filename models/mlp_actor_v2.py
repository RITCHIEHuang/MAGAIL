#!/usr/bin/env python
# Created at 2020/2/15
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from custom.MultiOneHotCategorical import MultiOneHotCategorical
from custom.MultiSoftMax import MultiSoftMax


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, num_discrete_actions=0, discrete_actions_sections: Tuple = (0,),
                 action_log_std=0, use_multivariate_distribution=False,
                 num_hiddens: Tuple = (64, 64), activation: nn.Module = nn.LeakyReLU,
                 drop_rate=None):
        """
        Deal with hybrid of discrete actions and continuous actions,
        if there's discrete actions, we put discrete actions at left, and continuous actions on the right.
        That's say, each action is arranged by the follow form :
            action = [discrete_action, continuous_action]

        :param num_states:
        :param num_actions:
        :param num_discrete_actions: OneHot encoded actions
        :param discrete_actions_sections:
        :param action_log_std:
        :param num_hiddens:
        :param activation:
        :param drop_rate:
        """
        super(Actor, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.use_multivariate_distribution = use_multivariate_distribution
        # set up discrete action info
        self.num_discrete_actions = num_discrete_actions
        assert sum(discrete_actions_sections) == num_discrete_actions, f"Expected sum of discrete actions's " \
                                                                       f"dimension =  {num_discrete_actions}"
        self.discrete_action_sections = discrete_actions_sections

        # set up continuous action log_std
        self.action_log_std = nn.Parameter(torch.ones(1, self.num_actions - self.num_discrete_actions) * action_log_std,
                                           requires_grad=True)

        # set up module units
        _module_units = [num_states]
        _module_units.extend(num_hiddens)
        _module_units += num_actions,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            n_units_in, n_units_out = module_unit
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
                self._module_list.add_module(f"Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        self._module_list.add_module(f"Layer_{idx + 1}_Activation", nn.Tanh())
        # if there's discrete actions, add custom Soft Max layer
        if self.num_discrete_actions:
            self._module_list.add_module(f"Layer_{idx + 1}_Custom_Softmax",
                                         MultiSoftMax(0, self.num_discrete_actions, self.discrete_action_sections))

    def forward(self, x):
        """
        give states, calculate the distribution of actions
        :param x: unsqueezed states
        :return: xxx
        """
        for module in self._module_list:
            x = module(x)
        # note that x include [discrete_action_softmax probability, continuous_action_mean]
        # extract discrete_action probs
        dist_discrete = None
        if self.num_discrete_actions:
            dist_discrete = MultiOneHotCategorical(x[..., :self.num_discrete_actions],
                                                   sections=self.discrete_action_sections)
        continuous_action_mean = x[..., self.num_discrete_actions:]
        continuous_action_log_std = self.action_log_std.expand_as(x[..., self.num_discrete_actions:])
        continuous_action_std = torch.exp(continuous_action_log_std)

        if self.use_multivariate_distribution:
            dist_continuous = MultivariateNormal(continuous_action_mean, torch.diag_embed(continuous_action_std))
        else:
            dist_continuous = Normal(continuous_action_mean, continuous_action_std)

        return dist_discrete, dist_continuous

    def get_action_log_prob(self, states):
        """
        give states, select actions based on the distribution
        and calculate the log probability of actions
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: actions and log probablities
        """
        dist_discrete, dist_continuous = self.forward(states)
        action = dist_continuous.sample()  # [batch_size, num_actions]
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(action)  # use multivariate normal distribution
        else:
            log_prob = dist_continuous.log_prob(action).sum(dim=-1)  # [batch_size]

        if dist_discrete:
            discrete_action = dist_discrete.sample()
            discrete_log_prob = dist_discrete.log_prob(discrete_action)  # [batch_size]
            action = torch.cat([discrete_action, action], dim=-1)

            """
            How to deal with log prob?
            
            1. Add discrete log_prob and continuous log_prob, consider their dependency;
            2. Concat them together
            """
            log_prob = (log_prob + discrete_log_prob)  # add log prob [batch_size, 1]
            # log_prob = torch.cat([discrete_log_prob, log_prob], dim=-1)  # concat [batch_size, 2]

        log_prob.unsqueeze_(-1)
        return action, log_prob  # log_prob [batch_size, 1/2]

    def get_log_prob(self, states, actions):
        """
        give states and actions, calculate the log probability
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: log probabilities
        """
        dist_discrete, dist_continuous = self.forward(states)
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:])
        else:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:]).sum(dim=-1)
        if dist_discrete:
            discrete_log_prob = dist_discrete.log_prob(actions[..., :self.num_discrete_actions])
            log_prob = log_prob + discrete_log_prob
        return log_prob.unsqueeze(-1)

    def get_entropy(self, states):
        """
        give states, calculate the entropy of actions' distribution
        :param states: unsqueezed states
        :return: mean entropy
        """
        dist_discrete, dist_continuous = self.forward(states)
        ent_discrete = dist_discrete.entropy()
        ent_continuous = dist_continuous.entropy()

        ent = torch.cat([ent_discrete, ent_continuous], dim=-1).unsqueeze_(-1)  # [batch_size, 2]
        return ent

    def get_kl(self, states):
        """
        give states, calculate the KL_Divergence of actions' distribution
        :param states: unsqueezed states
        :return: mean kl divergence
        """
        pass
