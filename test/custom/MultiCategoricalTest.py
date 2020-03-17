#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/17 下午12:57
import unittest

import torch
from torch.distributions import Categorical

from custom.MultiCategorical import MultiCategorical


class MultiCategoricalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_probs = torch.tensor([[0.3, 0.2, 0.4, 0.1, 0.25, 0.5, 0.25, 0.3, 0.4, 0.1, 0.1, 0.1],
                                        [0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1]])
        self.test_sections = (4, 3, 5)

        self.test_actions = torch.tensor([[2, 1, 1],
                                          [3, 0, 2]]).long()

        self.test_sected_actions = torch.split(self.test_actions, 1, dim=-1)

        self.test_multi_categorical = MultiCategorical(self.test_probs, self.test_sections)

        self.test_categorical1 = Categorical(self.test_probs[:, :4])
        self.test_categorical2 = Categorical(self.test_probs[:, 4:7])
        self.test_categorical3 = Categorical(self.test_probs[:, 7:])

    def test_log_prob(self):
        test_cat1_log_prob = self.test_categorical1.log_prob(self.test_sected_actions[0].squeeze())
        test_cat2_log_prob = self.test_categorical2.log_prob(self.test_sected_actions[1].squeeze())
        test_cat3_log_prob = self.test_categorical3.log_prob(self.test_sected_actions[2].squeeze())

        test_multi_cat_log_prob = self.test_multi_categorical.log_prob(self.test_actions)
        print(test_multi_cat_log_prob)
        print(test_cat1_log_prob)
        self.assertEqual(test_cat1_log_prob.shape, test_multi_cat_log_prob.shape)

        self.assertTrue(
            torch.equal(test_cat1_log_prob + test_cat2_log_prob + test_cat3_log_prob, test_multi_cat_log_prob))

    def test_sample(self):
        test_cat1_sample = self.test_categorical1.sample()
        test_cat2_sample = self.test_categorical2.sample()
        test_cat3_sample = self.test_categorical3.sample()

        test_cat_sample = torch.cat(
            [test_cat1_sample.unsqueeze(-1), test_cat2_sample.unsqueeze(-1), test_cat3_sample.unsqueeze(-1)], dim=-1)
        test_multi_cat_sample = self.test_multi_categorical.sample()

        self.assertEqual(test_cat_sample.shape, test_multi_cat_sample.shape)

    def test_entropy(self):
        test_cat1_entropy = self.test_categorical1.entropy()
        test_cat2_entropy = self.test_categorical2.entropy()
        test_cat3_entropy = self.test_categorical3.entropy()

        test_multi_cat_entropy = self.test_multi_categorical.entropy()

        self.assertTrue(torch.equal(test_cat1_entropy + test_cat2_entropy + test_cat3_entropy, test_multi_cat_entropy),
                        "Expected same entropy!!!")


if __name__ == '__main__':
    unittest.main()
