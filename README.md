# Multi-Agent Generative Adversarial Imitation Learning

This repo contains a general `MAGAIL` implementation, it's useful when learning a `Joint-policy`: 
including `agent policy and environment policy`.  
The `Agent` can interact with `Environment` by taking action according to the state given by the `Environment`, 
and the `Environment` decide to send state according to agent's action.

## 1.Formulation of Joint-policy

In [GAIL](https://arxiv.org/pdf/1606.03476.pdf), it's the most trivial case that only a `single` agent. `Multi-Agent` can be general more than one, here we only focus two agents.  
As you can imagine a scenario in `commodity recommendation`:  The Platform will decide what kind of commodities to recommend according to user's action(buy ? browse ? search ? add to shopping cart ? ......);
From another point of view, A user will take corresponding actions according to what they see(he like the goods, so he bought; he is interested in the commodities, so he browse them or add to the shopping cart). 

The structure should be like this:

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcxag8vihbj315c0c7dg4.jpg)


## 2.Usage

1. To run the example, first you need to install necessary dependencies:
```textmate
1. python >= 3.6
2. pytorch >= 1.3.1
3. pandas >= 1.0.1
4. PyYAML >= 5.3    
```
2. Filling in model parameters into [config/config.yml](config/config.yml)

An example configuration file should be like this:
```yaml
# general parameters
general:
  seed: 2020
  expert_batch_size: 2000
  training_epochs: 1000000

# parameters for general advantage estimation
gae:
  gamma: 0.995
  tau: 0.96

# parameters for PPO algorithm
ppo:
  clip_ratio: 0.1
  ppo_optim_epochs: 1
  ppo_mini_batch_size: 128
  sample_batch_size: 5000

# parameters for joint-policy
policy:
  learning_rate: !!float 3e-4
  trajectory_length: 10
  num_states: 155
  num_actions: 6
  user:
    num_states: 155
    num_actions: 6
    num_discrete_actions: 0
    discrete_actions_sections: !!python/tuple [0]
    action_log_std: 0.0
    num_hiddens: !!python/tuple [256, 256]
    activation: nn.LeaklyReLU
    drop_rate: 0.5
  env:
    num_states: 161
    num_actions: 155
    num_discrete_actions: 132
    discrete_actions_sections: !!python/tuple [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
    action_log_std: 0.0
    num_hiddens: !!python/tuple [256, 256]
    activation: nn.LeaklyReLU
    drop_rate: 0.5

# parameters for critic
value:
  num_states: 155
  num_hiddens: !!python/tuple [256, 256]
  activation: nn.LeaklyReLU
  drop_rate: 0.5
  learning_rate: !!float 3e-3
  l2_reg: !!float 1e-3

# parameters for discriminator
discriminator:
  num_states: 155
  num_actions: 6
  num_hiddens: !!python/tuple [256, 256]
  activation: nn.LeaklyReLU
  drop_rate: 0.5
  learning_rate: !!float 1e-4
```

## 3.Performance

For judging performance of the algorithm, we mainly focus on the `Reward` given by `Discriminator`. You may need fine tuning
in your experiments, luckily, almost all the tips and tricks applying to [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) can be used in `MAGAIL` training.

1. Discriminator Loss
![Discriminator Loss](https://tva1.sinaimg.cn/large/00831rSTgy1gcxb7seq1yj30wh0bg74k.jpg)

2. Expert Reward
![Expert Reward](https://tva1.sinaimg.cn/large/00831rSTgy1gcxbe1hkq4j30wd0bkjrs.jpg)

3. Generator Reward
![Generator Reward](https://tva1.sinaimg.cn/large/00831rSTgy1gcxbewooufj30w90bimxi.jpg)

As you can see, the algorithm is not perfect meet our expectation.Any how, it's Reinforcement Learning at all, just keep trying!!!

## 4.Reference

\[1\]. [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf))
\[2\]. [Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning](https://arxiv.org/pdf/1805.10000.pdf)
\[3\]. [Tricks of GANs](https://lanpartis.github.io/deep%20learning/2018/03/12/tricks-of-gans.html)