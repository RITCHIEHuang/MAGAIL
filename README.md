# Multi-Agent Generative Adversarial Imitation Learning

   This repo contains a general `MAGAIL` implementation, it's useful when learning a **Joint-policy** : 
mixing `agent policy` and `environment policy` together.  The `Agent` can interact with `Environment` by taking action according to the state given by the `Environment`, 
and the `Environment` sends state according to agent's action.
   

## 1.Formulation of Joint-policy

   In [GAIL](https://arxiv.org/pdf/1606.03476.pdf), it's the most trivial case that only a **single** agent. Multi-Agent can be general more than one, here we only focus two agents.  

   As you can imagine a scenario in `Commodity Recommendation` :  The Platform will decide what kind of commodities to recommend according to user's action (buy ? browse ? search ? add to shopping cart ? ...... ),
from another point of view, A user will take corresponding actions according to what they see(he like the goods, so he bought, he is interested in the commodities, so he browse them or add to the shopping cart). 

The structure should be like this:

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcxag8vihbj315c0c7dg4.jpg)


## 2.Usage

### 1. To run the example, first you need to install necessary dependencies:

```textmate
1. python >= 3.6
2. pytorch >= 1.3.1
3. pandas >= 1.0.1
4. PyYAML >= 5.3    
```
    
### 2. Filling in model parameters into [config/config.yml](config/config.yml)

An example configuration file should be like this:

```yaml
# general parameters
general:
  seed: 2020
  expert_batch_size: 2000
  expert_data_path: ../data/train_data_sas.csv
  training_epochs: 500000
  num_states: 155
  num_actions: 6

# parameters for general advantage estimation
gae:
  gamma: 0.995
  tau: 0.96

# parameters for PPO algorithm
ppo:
  clip_ratio: 0.1
  ppo_optim_epochs: 1
  ppo_mini_batch_size: 200
  sample_batch_size: 2000

# parameters for joint-policy
jointpolicy:
  learning_rate: !!float 1e-4
  trajectory_length: 10
  user:
    num_states: 155
    num_actions: 6
    num_discrete_actions: 0
    discrete_actions_sections: !!python/tuple [0]
    action_log_std: 0.0
    use_multivariate_distribution: False
    num_hiddens: !!python/tuple [256]
    activation: LeaklyReLU
    drop_rate: 0.5
  env:
    num_states: 161
    num_actions: 155
    num_discrete_actions: 132
    discrete_actions_sections: !!python/tuple [5, 2, 4, 3, 2, 9, 2, 32, 35, 7, 2, 21, 2, 3, 3]
    action_log_std: 0.0
    use_multivariate_distribution: False
    num_hiddens: !!python/tuple [256]
    activation: LeakyReLU
    drop_rate: 0.5

# parameters for critic
value:
  num_states: 155
  num_hiddens: !!python/tuple [256, 256]
  activation: LeakyReLU
  drop_rate: 0.5
  learning_rate: !!float 3e-4
  l2_reg: !!float 1e-3

# parameters for discriminator
discriminator:
  num_states: 155
  num_actions: 6
  num_hiddens: !!python/tuple [256, 256]
  activation: LeakyReLU
  drop_rate: 0.5
  learning_rate: !!float 4e-4
  use_noise: True # trick: add noise
  noise_std: 0.15
  use_label_smoothing: True # trick: label smoothing
  label_smooth_rate: 0.1

```

## 3.Performance

   For judging performance of the algorithm, we mainly focus on the `Reward` given by `Discriminator`. You may need fine tuning
in your experiments, luckily, almost all the tips and tricks applying to [GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) can be used in `MAGAIL` training.

1. Discriminator Loss

It's identical to original GAN's discriminator loss objective:
$$
    - \mathbb E_{x \sim p_{expert}} \left[\log D(x)\right] - \mathbb E_{x \sim p_{generated}} \left[ \log (1 - D(x)) \right]
$$

The optimal result will be asymptotic to $2 \log 2 \simeq 1.3862....$ while it's about 1.17 here : (
 
![Discriminator Loss](https://tva1.sinaimg.cn/large/007S8ZIlly1ged6l1jgvgj31d20p2js9.jpg)

2. Expert Reward(Discriminator output)

At the beginning, the discriminator can easily tell which is from expert and assign a high reward which can be about 0.97, 
As the Policy improve gradually, it starts to go down and eventually converges to around 0.6.

![Expert's Reward](https://tva1.sinaimg.cn/large/007S8ZIlgy1ged6lxfcyrj31ck0oodgp.jpg)

3. Generator Reward(Discriminator output)

In Generator, it just acts like the opposite, and finally converges to about 0.4.

![Generator's Reward](https://tva1.sinaimg.cn/large/007S8ZIlgy1ged6mm6z1mj31cm0oomy8.jpg)

The final converge ratio is about 6:4 (you may see some slight tendency to break out this convergence, just train it for much longer time) , 
which is not perfect but usable (As you may know that training GAN is so hard needless to say MAGAIL).  


## 4. Build environment based on the pre-trained policy in RealWorld

From step 1 to 3, the policy will be trained fine and then it should be used in real world cases.

## 5.Reference

\[1\]. [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf)    
\[2\]. [Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning](https://arxiv.org/pdf/1805.10000.pdf)  
\[3\]. [Tricks of GANs](https://lanpartis.github.io/deep%20learning/2018/03/12/tricks-of-gans.html)  
\[4\]. [IMPROVING GENERATIVE ADVERSARIAL IMITATION LEARNING WITH NON-EXPERT DEMONSTRATIONS](https://openreview.net/pdf?id=BJl65sA9tm)  
\[5\]. [Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB)  
\[6\]. [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://openreview.net/forum?id=S1xKd24twB)  