#!/usr/bin/env python
# Created at 2020/5/7
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from custom.MultiOneHotCategorical import MultiOneHotCategorical
from custom.MultiSoftMax import MultiSoftMax
from utils.torch_util import resolve_activate_function


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, num_discrete_actions=0, discrete_actions_sections: Tuple = (0,),
                 action_limit=1, action_log_std_min=-20, action_log_std_max=2, use_multivariate_distribution=False,
                 num_hiddens: Tuple = (128, 128), activation: str = "relu", drop_rate=None):

        super(Actor, self).__init__()
        # set up state space and action space

        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.action_limit = action_limit
        self.action_log_std_min = action_log_std_min
        self.action_log_std_max = action_log_std_max

        self.use_multivariate_distribution = use_multivariate_distribution
        # set up discrete action info
        self.num_discrete_actions = num_discrete_actions
        assert num_discrete_actions == 0, f"Expected no discrete action, but get {num_discrete_actions}!!!"
        assert sum(discrete_actions_sections) == num_discrete_actions, f"Expected sum of discrete actions's " \
                                                                       f"dimension =  {num_discrete_actions}"
        self.discrete_action_sections = discrete_actions_sections

        # set up continuous action log_std
        self.action_log_std = nn.Linear(num_hiddens[-1], self.num_actions - self.num_discrete_actions)
        self.action_mean = nn.Linear(num_hiddens[-1], num_actions)

        # set up module units
        _module_units = [num_states]
        _module_units.extend(num_hiddens)

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

        # set up hidden layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            n_units_in, n_units_out = module_unit
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
                self._module_list.add_module(f"Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        # if there's discrete actions, add custom Soft Max layer
        if self.num_discrete_actions:
            self.custom_softmax = MultiSoftMax(0, self.num_discrete_actions, self.discrete_action_sections)

    def forward(self, x):
        """
        give states, calculate the distribution of actions
        :param x: unsqueezed states
        :return: xxx
        """
        for module in self._module_list:
            x = module(x)

        continuous_action_log_std = self.action_log_std(x)  # [batch_size, num_continuous_actions]
        x = self.action_mean(x)  # [batch_size, num_actions]

        # note that x include [discrete_action_softmax probability, continuous_action_mean]
        # extract discrete_action probs
        dist_discrete = None
        if self.num_discrete_actions:
            x = self.custom_softmax(x)
            dist_discrete = MultiOneHotCategorical(x[..., :self.num_discrete_actions],
                                                   sections=self.discrete_action_sections)
        continuous_action_mean = x[..., self.num_discrete_actions:]

        continuous_action_log_std = continuous_action_log_std.clamp_(self.action_log_std_min, self.action_log_std_max)
        continuous_action_std = torch.exp(continuous_action_log_std)

        if self.use_multivariate_distribution:
            dist_continuous = MultivariateNormal(continuous_action_mean, torch.diag_embed(continuous_action_std))
        else:
            dist_continuous = Normal(continuous_action_mean, continuous_action_std)

        return dist_discrete, dist_continuous

    def get_action_log_prob(self, states, eps=1e-6):

        dist_discrete, dist_continuous = self.forward(states)
        action = dist_continuous.sample()  # [batch_size, num_actions]
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(action)  # use multivariate normal distribution
        else:
            log_prob = dist_continuous.log_prob(action).sum(dim=-1)  # [batch_size]

        action = torch.tanh(action)  # [batch_size, num_actions]
        log_prob -= (torch.log(1. - action.pow(2) + eps)).sum(dim=-1)  # [batch_size]

        if dist_discrete:
            discrete_action = dist_discrete.sample()
            discrete_log_prob = dist_discrete.log_prob(discrete_action)  # [batch_size]
            action = torch.cat([discrete_action, action], dim=-1)

            log_prob = (log_prob + discrete_log_prob)  # add log prob [batch_size, 1]
            # log_prob = torch.cat([discrete_log_prob, log_prob], dim=-1)  # concat [batch_size, 2]

        return action * self.action_limit, log_prob  # log_prob [batch_size, 1/2]
