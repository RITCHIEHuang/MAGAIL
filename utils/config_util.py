#!/usr/bin/env python
# Created at 2020/5/1
import yaml

from utils.time_util import timer


@timer(message="Loading model configuration !", show_result=True)
def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config
