#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created at 2020/1/17 下午12:57
import unittest

import torch
from torch.distributions import OneHotCategorical

from custom.MultiOneHotCategorical import MultiOneHotCategorical


class MultiOneHotCategoricalTest(unittest.TestCase):

    def setUp(self) -> None:
        self.test_probs = torch.tensor([[0.3, 0.2, 0.4, 0.1, 0.25, 0.5, 0.25, 0.3, 0.4, 0.1, 0.1, 0.1],
                                        [0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1]])
        self.test_sections = (4, 3, 5)

        self.test_actions = torch.tensor([[0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
                                          [0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.]]).long()

        self.test_sected_actions = torch.split(self.test_actions, self.test_sections, dim=-1)

        self.test_multi_onehot_categorical = MultiOneHotCategorical(self.test_probs, self.test_sections)

        self.test_onehot_categorical1 = OneHotCategorical(self.test_probs[:, :4])
        self.test_onehot_categorical2 = OneHotCategorical(self.test_probs[:, 4:7])
        self.test_onehot_categorical3 = OneHotCategorical(self.test_probs[:, 7:])

    def test_log_prob(self):
        test_cat1_log_prob = self.test_onehot_categorical1.log_prob(self.test_sected_actions[0])
        test_cat2_log_prob = self.test_onehot_categorical2.log_prob(self.test_sected_actions[1])
        test_cat3_log_prob = self.test_onehot_categorical3.log_prob(self.test_sected_actions[2])

        test_multi_cat_log_prob = self.test_multi_onehot_categorical.log_prob(self.test_actions)
        print(test_multi_cat_log_prob)
        print(test_cat1_log_prob)
        self.assertEqual(test_cat1_log_prob.shape, test_multi_cat_log_prob.shape)

        self.assertTrue(
            torch.equal(test_cat1_log_prob + test_cat2_log_prob + test_cat3_log_prob, test_multi_cat_log_prob))

    def test_sample(self):
        test_cat1_sample = self.test_onehot_categorical1.sample()
        test_cat2_sample = self.test_onehot_categorical2.sample()
        test_cat3_sample = self.test_onehot_categorical3.sample()

        test_cat_sample = torch.cat([test_cat1_sample, test_cat2_sample, test_cat3_sample], dim=-1)
        test_multi_cat_sample = self.test_multi_onehot_categorical.sample()

        self.assertEqual(test_cat_sample.shape, test_multi_cat_sample.shape)
        self.assertTrue(torch.equal(test_cat_sample.sum(dim=-1),
                                    test_multi_cat_sample.sum(dim=-1)))

    def test_entropy(self):
        test_cat1_entropy = self.test_onehot_categorical1.entropy()
        test_cat2_entropy = self.test_onehot_categorical2.entropy()
        test_cat3_entropy = self.test_onehot_categorical3.entropy()

        test_multi_cat_entropy = self.test_multi_onehot_categorical.entropy()

        self.assertTrue(torch.equal(test_cat1_entropy + test_cat2_entropy + test_cat3_entropy, test_multi_cat_entropy),
                        "Expected same entropy!!!")


if __name__ == '__main__':
    unittest.main()
