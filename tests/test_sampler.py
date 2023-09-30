# -*- coding: utf-8 -*-

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

from unittest import TestCase

import numpy as np
import pandas as pd

from src.tabgan.sampler import OriginalGenerator, Sampler, GANGenerator, ForestDiffusionGenerator


class TestOriginalGenerator(TestCase):
    def test_get_object_generator(self):
        gen = OriginalGenerator(gen_x_times=15)
        self.assertTrue(isinstance(gen.get_object_generator(), Sampler))


class TestGANGenerator(TestCase):
    def test_get_object_generator(self):
        gen = GANGenerator(gen_x_times=15)
        self.assertTrue(isinstance(gen.get_object_generator(), Sampler))


class TestSamplerOriginal(TestCase):
    def setUp(self):
        self.train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list('ABCD'))
        self.target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list('Y'))
        self.test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        self.gen = OriginalGenerator(gen_x_times=15)
        self.sampler = self.gen.get_object_generator()

    def test_preprocess_data(self):
        self.setUp()
        new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                      self.target.copy(), self.test)
        self.assertEqual(self.test.shape, test_df.shape)
        self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
        self.assertEqual(new_train.shape[0], new_target.shape[0])

        self.assertTrue(isinstance(new_train, pd.DataFrame))
        self.assertTrue(isinstance(new_target, pd.DataFrame))
        self.assertTrue(isinstance(test_df, pd.DataFrame))
        args = [self.train.head(), self.target.copy(), self.test.to_numpy()]
        self.assertRaises(ValueError, self.sampler.preprocess_data, *args)

    def test_generate_data(self):
        new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                      self.target.copy(), self.test)
        gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df, only_generated_data=False)
        self.assertEqual(gen_train.shape[0], gen_target.shape[0])
        self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
        self.assertTrue(gen_train.shape[0] > new_train.shape[0])
        self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))

    def test_postprocess_data(self):
        new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                      self.target.copy(), self.test)
        gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df, only_generated_data=False)
        new_train, new_target = self.sampler.postprocess_data(gen_train, gen_target, test_df)
        self.assertEqual(new_train.shape[0], new_target.shape[0])
        self.assertGreaterEqual(new_train.iloc[:, 0].min(), test_df.iloc[:, 0].min())
        self.assertGreaterEqual(test_df.iloc[:, 0].max(), new_train.iloc[:, 0].max())

    def test_adversarial_filtering(self):
        new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                      self.target.copy(), self.test)
        gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df, only_generated_data=False)
        new_train, new_target = self.sampler.postprocess_data(gen_train, gen_target, test_df)
        new_train, new_target = self.sampler.adversarial_filtering(new_train, new_target, test_df)
        self.assertEqual(new_train.shape[0], new_target.shape[0])

    def test__validate_data(self):
        result = self.sampler._validate_data(self.train.copy(), self.target.copy(), self.test)
        self.assertIsNone(result)
        args = [self.train.head(), self.target.copy(), self.test]
        self.assertRaises(ValueError, self.sampler._validate_data, *args)

    class TestSamplerGAN(TestCase):
        def setUp(self):
            self.train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list('ABCD'))
            self.target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list('Y'))
            self.test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
            self.gen = GANGenerator(gen_x_times=15)
            self.sampler = self.gen.get_object_generator()

        def test_generate_data(self):
            new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                          self.target.copy(), self.test)
            gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df)
            self.assertEqual(gen_train.shape[0], gen_target.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
            self.assertTrue(gen_train.shape[0] > new_train.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))

    class TestSamplerGAN(TestCase):
        def setUp(self):
            self.train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list('ABCD'))
            self.target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list('Y'))
            self.test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
            self.gen = ForestDiffusionGenerator(gen_x_times=15)
            self.sampler = self.gen.get_object_generator()

        def test_generate_data(self):
            new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                          self.target.copy(), self.test)
            gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df)
            self.assertEqual(gen_train.shape[0], gen_target.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
            self.assertTrue(gen_train.shape[0] > new_train.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
