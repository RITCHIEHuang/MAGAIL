#!/usr/bin/env python
# Created at 2020/3/14
import click
import yaml

from algos.MAGAIL import MAGAIL
from utils.time_util import timer


@click.command()
@click.option("--config_path", type=str, default="./config/config.yml",
              help="Path for model configuration")
@click.option("--eval_model_epoch", type=int, default=50, help="Intervals for evaluating model")
@click.option("--save_model_epoch", type=int, default=50, help="Intervals for saving model")
@click.option("--save_model_path", type=str, default="./model_pkl", help="Path for saving trained model")
@click.option("--load_model", type=bool, default=False, help="Indicator for whether load trained model")
@click.option("--load_model_path", type=str, default=None, help="Path for loading trained model")
def main(config_path, eval_model_epoch, save_model_epoch, save_model_path, load_model,
         load_model_path):
    config = config_loader(path=config_path)  # load model configuration
    training_epochs = config["general"]["training_epochs"]

    mail = MAGAIL(config=config, log_dir="./log")

    if load_model:
        mail.load_model(load_model_path)

    for epoch in range(1, training_epochs + 1):
        mail.train(epoch)

        if epoch % eval_model_epoch == 0:
            mail.eval(epoch)

        if epoch % save_model_epoch == 0:
            mail.save_model(save_model_path)


@timer(message="Loading model configuration !", show_result=False)
def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


if __name__ == '__main__':
    main()
