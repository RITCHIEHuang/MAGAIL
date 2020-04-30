#!/usr/bin/env python
# Created at 2020/2/15
# !/usr/bin/env python
# Created at 2020/2/15
from typing import Tuple

import torch
import torch.nn as nn

from utils.torch_utils import device


class Discriminator(nn.Module):
    def __init__(self, num_states, num_actions, num_hiddens: Tuple = (64, 64), activation: nn.Module = nn.LeakyReLU,
                 drop_rate=None, use_noise=False, noise_std=0.1):
        super(Discriminator, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.num_value = 1
        # set up module units
        _module_units = [num_states + num_actions]
        _module_units.extend(num_hiddens)
        _module_units += self.num_value,
        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]

        # set up module layers
        self._module_list = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units):
            self._module_list.add_module(f"Layer_{idx + 1}_Linear", nn.Linear(*module_unit))
            if idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Activation", activation())
            if self.drop_rate and idx != len(self._layers_units) - 1:
                self._module_list.add_module(f"Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))
        self._module_list.add_module(f"Layer_{idx + 1}_Activation", nn.Sigmoid())

    def forward(self, states, actions):
        """
        give states, calculate the estimated values
        :param states: unsqueezed states
        :param actions: unsqueezed actions
        :return: values
        """
        x = torch.cat([states, actions], dim=-1)
        if self.use_noise:  # trick: add gaussian noise to discriminator
            x += torch.normal(0, self.noise_std, size=x.shape, device=device)
        for module in self._module_list:
            x = module(x)
        return x
