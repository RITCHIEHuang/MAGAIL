#!/usr/bin/env python
# Created at 2020/3/14
import click
import time
from tqdm import tqdm
from magail.MAGAIL import MAGAIL
from utils.config_util import config_loader


@click.command()
@click.option("--train_mode", type=bool, default=False, help="Train / Validate")
@click.option("--eval_model_epoch", type=int, default=1000, help="Intervals for evaluating model")
@click.option("--save_model_epoch", type=int, default=1000, help="Intervals for saving model")
@click.option("--save_model_path", type=str, default="../model_pkl", help="Path for saving trained model")
@click.option("--load_model", type=bool, default=False, help="Indicator for whether load trained model")
@click.option("--load_model_path", type=str, default="../model_pkl/MAGAIL_Train_2020-05-01_18:09:33", help="Path for loading trained model")
def main(train_mode, eval_model_epoch, save_model_epoch, save_model_path, load_model,
         load_model_path):

    if train_mode:
        config_path = "../config/config.yml"

        exp_name = f"MAGAIL_Train_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"
    else:
        config_path = "../config/config_validation.yml"
        exp_name = f"MAGAIL_Validate_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"

    config = config_loader(path=config_path)  # load model configuration
    training_epochs = config["general"]["training_epochs"]

    mail = MAGAIL(config=config, log_dir="../log", exp_name=exp_name)

    if load_model:
        print(f"Loading Pre-trained MAGAIL model from {load_model_path}!!!")
        mail.load_model(load_model_path)

    for epoch in tqdm(range(1, training_epochs + 1)):
        mail.train(epoch)

        if epoch % eval_model_epoch == 0:
            mail.eval(epoch)

        if epoch % save_model_epoch == 0:
            mail.save_model(save_model_path)


if __name__ == '__main__':
    main()
