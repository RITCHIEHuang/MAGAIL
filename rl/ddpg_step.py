#!/usr/bin/env python
# Created at 2020/1/22
import torch
import torch.nn as nn


def soft_update(target, source, polyak):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              states, actions, rewards, next_states, masks, gamma, polyak):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    """update critic"""

    values = value_net(states, actions)

    with torch.no_grad():
        target_next_values = value_net_target(next_states, policy_net_target(next_states))
        target_values = rewards + gamma * masks * target_next_values
    value_loss = nn.MSELoss()(values, target_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update actor"""

    policy_loss = - value_net(states, policy_net(states)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """soft update target nets"""
    soft_update(policy_net_target, policy_net, polyak)
    soft_update(value_net_target, value_net, polyak)
    return value_loss, policy_loss
