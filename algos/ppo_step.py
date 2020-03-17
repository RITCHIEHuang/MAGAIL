#!/usr/bin/env python
# Created at 2020/3/10
import torch
import torch.nn as nn


def ppo_step(policy_net, value_net, optimizer_p, optimizer_v, states, actions, next_states, returns, old_log_probs,
             advantages, ppo_clip_ratio, value_l2_reg):
    # update value net
    values_pred = value_net(states)
    value_loss = nn.MSELoss()(values_pred, returns)

    for param in value_net.parameters():
        value_loss += value_l2_reg * param.pow(2).sum()

    optimizer_v.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 40)
    optimizer_v.step()

    # update policy net
    new_log_probs = policy_net.get_log_prob(states, actions, next_states)
    ratio = torch.exp(new_log_probs - old_log_probs)

    sur_loss1 = ratio * advantages
    sur_loss2 = torch.clamp(ratio, 1 - ppo_clip_ratio, 1 + ppo_clip_ratio) * advantages
    policy_loss = -torch.min(sur_loss1, sur_loss2).mean()

    optimizer_p.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_p.step()

    return value_loss, policy_loss
