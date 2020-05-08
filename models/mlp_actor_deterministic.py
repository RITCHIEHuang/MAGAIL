#!/usr/bin/env python
# Created at 2020/5/6
from typing import Tuple

import torch.nn as nn

from utils.torch_util import resolve_activate_function


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, action_limit, num_hiddens: Tuple = (128, 128), activation: str = "relu",
                 drop_rate=None):
        """
        Deal with deterministic policy using by DDPG
        :param num_states:
        :param num_actions:
        :param num_hiddens:
        :param activation:
        :param drop_rate:
        """
        super(Actor, self).__init__()
        # set up state space and action space
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_limit = action_limit
        self.drop_rate = drop_rate

        # set up module units
        _module_units = [num_states]
        _module_units.extend(num_hiddens)
        _module_units += num_actions,

        self._layers_units = [(_module_units[i], _module_units[i + 1]) for i in range(len(_module_units) - 1)]
        activation = resolve_activate_function(activation)

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

    def forward(self, x):
        """
        give states, return deterministic action
        :param x: unsqueezed states
        :return: xxx
        """
        for module in self._module_list:
            x = module(x)

        return x * self.action_limit
