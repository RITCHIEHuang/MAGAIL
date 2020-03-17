#!/usr/bin/env python
# Created at 2020/2/15
# !/usr/bin/env python
# Created at 2020/2/15
from typing import Tuple

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, num_states, num_actions, num_hiddens: Tuple = (64, 64), activation: nn.Module = nn.LeakyReLU,
                 drop_rate=None):
        super(Discriminator, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.drop_rate = drop_rate
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
        self._module_list.append(nn.Sigmoid())

    def forward(self, states, actions):
        """
        give states, calculate the estimated values
        :param x: unsqueezed states
        :return: values
        """
        x = torch.cat([states, actions], dim=-1)
        for module in self._module_list:
            x = module(x)
        return x
