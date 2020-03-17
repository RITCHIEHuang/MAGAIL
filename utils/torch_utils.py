#!/usr/bin/env python
# Created at 2020/2/15
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

FLOAT = torch.FloatTensor
LONG = torch.LongTensor
DOUBLE = torch.DoubleTensor


def to_device(*params):
    return [x.to(device) for x in params]


def init_module(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)

