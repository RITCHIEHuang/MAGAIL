#!/usr/bin/env python
# Created at 2020/3/10
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.torch_utils import FLOAT


class ExpertDataSet(Dataset):
    def __init__(self, data_set_path, num_states, num_actions):
        self.expert_data = np.array(pd.read_csv(data_set_path))
        self.state = FLOAT(self.expert_data[:, :num_states])
        self.action = FLOAT(self.expert_data[:, num_states:num_states + num_actions])
        self.next_state = FLOAT(self.expert_data[:, num_states + num_actions:])
        self.length = self.state.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.state[idx], self.action[idx]
