#!/usr/bin/env python
# Created at 2020/3/27
import click
import torch
from torch.utils.tensorboard import SummaryWriter

from env.WebEyeEnv import WebEyeEnv
from rl.sac_alpha import SAC_Alpha


@click.command()
@click.option("--render", type=bool, default=False, help="Render environment or not")
@click.option("--num_process", type=int, default=1, help="Number of process to run environment")
@click.option("--lr_p", type=float, default=3e-4, help="Learning rate for Policy Net")
@click.option("--lr_a", type=float, default=3e-4, help="Learning rate for Temperature")
@click.option("--lr_q", type=float, default=3e-4, help="Learning rate for QValue Net")
@click.option("--gamma", type=float, default=0.95, help="Discount factor")
@click.option("--polyak", type=float, default=0.99,
              help="Interpolation factor in polyak averaging for target networks")
@click.option("--memory_size", type=int, default=1000000, help="Size of replay memory")
@click.option("--batch_size", type=int, default=128, help="Batch size")
@click.option("--min_update_step", type=int, default=1000, help="Minimum interacts for updating")
@click.option("--update_step", type=int, default=100, help="Steps between updating policy and critic")
@click.option("--max_iter", type=int, default=100000, help="Maximum iterations to run")
@click.option("--eval_iter", type=int, default=50, help="Iterations to evaluate the model")
@click.option("--save_iter", type=int, default=100, help="Iterations to save the model")
@click.option("--target_update_delay", type=int, default=1, help="Frequency for target QValue Net update")
@click.option("--model_path", type=str, default="trained_models", help="Directory to store model")
@click.option("--log_path", type=str, default="../log/", help="Directory to save logs")
@click.option("--seed", type=int, default=1, help="Seed for reproducing")
def main(render, num_process, lr_p, lr_a, lr_q, gamma, polyak, memory_size,
         batch_size, min_update_step, update_step, max_iter, eval_iter,
         save_iter, target_update_delay, model_path, log_path, seed):
    base_dir = log_path + "WebEye" + "/SAC_Alpha_exp{}".format(seed)
    writer = SummaryWriter(base_dir)
    env = WebEyeEnv(config_path="../config/config_webeye_env.yml",
                    model_path="../model_pkl/Magail_2020-04-30_10:35:30.pt"
                    )
    global_step = 0
    sac_alpha = SAC_Alpha(env,
                          render=render,
                          num_process=num_process,
                          memory_size=memory_size,
                          lr_p=lr_p,
                          lr_a=lr_a,
                          lr_q=lr_q,
                          gamma=gamma,
                          polyak=polyak,
                          batch_size=batch_size,
                          min_update_step=min_update_step,
                          update_step=update_step,
                          target_update_delay=target_update_delay,
                          seed=seed)

    for i_iter in range(1, max_iter + 1):
        sac_alpha.learn(writer, i_iter, global_step)

        if i_iter % eval_iter == 0:
            sac_alpha.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            sac_alpha.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
