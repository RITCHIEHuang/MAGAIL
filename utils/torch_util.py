#!/usr/bin/env python
# Created at 2020/2/15
import torch
import torch.nn as nn

__all__ = ['device', 'FLOAT', 'LONG', 'DOUBLE', 'to_device', 'init_module', 'get_flat_grad_params',
           'resolve_activate_function']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

FLOAT = torch.FloatTensor
LONG = torch.LongTensor
DOUBLE = torch.DoubleTensor


def to_device(*params):
    return [x.to(device) for x in params]


def init_module(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal(m.weight)
        # size = m.weight.size()
        # fan_out = size[0]
        # fan_in = size[1]
        # variance = np.sqrt(2.0 / (fan_in + fan_out))
        # m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)


def get_flat_grad_params(model: nn.Module):
    """
    get flatted grad of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])


def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh
