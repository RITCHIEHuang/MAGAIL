#!/usr/bin/env python
# Created at 2020/3/14
import click
import yaml
import time
from tqdm import tqdm
# from algos.MAGAIL import MAGAIL
from algos.MAGAIL_v2 import MAGAIL
from utils.time_util import timer


@click.command()
@click.option("--train_mode", type=bool, default=True, help="Train / Validate")
@click.option("--eval_model_epoch", type=int, default=1, help="Intervals for evaluating model")
@click.option("--save_model_epoch", type=int, default=1000, help="Intervals for saving model")
@click.option("--save_model_path", type=str, default="./model_pkl", help="Path for saving trained model")
@click.option("--load_model", type=bool, default=True, help="Indicator for whether load trained model")
@click.option("--load_model_path", type=str, default="./model_pkl/MAGAIL_Train_2020-05-01_17:24:46", help="Path for loading trained model")
def main(train_mode, eval_model_epoch, save_model_epoch, save_model_path, load_model,
         load_model_path):

    if train_mode:
        config_path = "./config/config_v2.yml"
        exp_name = f"MAGAIL_Train_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
    else:
        config_path = "./config/config_v2_validation.yml"
        exp_name = f"MAGAIL_Validate_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"

    config = config_loader(path=config_path)  # load model configuration
    training_epochs = config["general"]["training_epochs"]

    mail = MAGAIL(config=config, log_dir="./log", exp_name=exp_name)

    if load_model:
        mail.load_model(load_model_path)

    for epoch in tqdm(range(1, training_epochs + 1)):
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
