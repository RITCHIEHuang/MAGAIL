#!/usr/bin/env python
# Created at 2020/3/10
import torch

from utils.torch_util import FLOAT, device


def estimate_advantages(rewards, masks, values, gamma, tau, trajectory_length):
    """
    General advantage estimate
    :param rewards: [trajectory length * parallel size, 1]
    :param masks: [trajectory length * parallel size, 1]
    :param values: [trajectory length * parallel size, 1]
    :param gamma:
    :param tau:
    :param trajectory_length: the length of trajectory
    :return:
    """
    trans_shape_func = lambda x: x.reshape(trajectory_length, -1, 1)
    rewards = trans_shape_func(rewards)  # [trajectory length, parallel size, 1]
    masks = trans_shape_func(masks)  # [trajectory length, parallel size, 1]
    values = trans_shape_func(values)  # [trajectory length, parallel size, 1]

    deltas = FLOAT(rewards.size()).to(device)
    advantages = FLOAT(rewards.size()).to(device)

    # calculate advantages in parallel
    prev_value = torch.zeros((rewards.size(1), 1), device=device)
    prev_advantage = torch.zeros((rewards.size(1), 1), device=device)

    for i in reversed(range(rewards.size(0))):
        deltas[i, ...] = rewards[i, ...] + gamma * prev_value * masks[i, ...] - values[i, ...]
        advantages[i, ...] = deltas[i, ...] + gamma * tau * prev_advantage * masks[i, ...]

        prev_value = values[i, ...]
        prev_advantage = advantages[i, ...]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    # reverse shape for ppo
    return advantages.reshape(-1, 1), returns.reshape(-1, 1)  # [trajectory length * parallel size, 1]
