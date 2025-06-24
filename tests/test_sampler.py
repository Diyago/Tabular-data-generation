# -*- coding: utf-8 -*-

__author__ = "Insaf Ashrapov"
__copyright__ = "Insaf Ashrapov"
__license__ = "Apache 2.0"

from unittest import TestCase
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd
from src.tabgan.sampler import OriginalGenerator, Sampler, GANGenerator, ForestDiffusionGenerator, LLMGenerator


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


class TestSamplerLLMConditional(TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            "Name": ["Anna", "Maria", "Ivan", "Sergey"],
            "Gender": ["F", "F", "M", "M"],
            "Age": [25, 30, 35, 40],
            "Occupation": ["Engineer", "Doctor", "Artist", "Teacher"]
        })
        self.target_df = pd.DataFrame({"Y": [0, 1, 0, 1]}) # Can be None if not used for LLM
        # test_df is used for postprocessing, might not be strictly needed for all LLM tests if postprocessing is off
        self.test_df = pd.DataFrame({
            "Name": ["Olga", "Boris", "Svetlana"],
            "Gender": ["F", "M", "F"],
            "Age": [28, 32, 45],
            "Occupation": ["Manager", "Pilot", "Scientist"]
        })

        # Default gen_params for LLMGenerator
        self.gen_params = {"batch_size": 32, "epochs": 1, "llm": "distilgpt2", "max_length": 50}


    @patch('tabgan.sampler.GReaT') # Mock GReaT where it's imported in sampler.py
    @patch.object(LLMGenerator, '_generate_via_prompt')
    def test_conditional_generation_basic(self, mock_generate_prompt, mock_great_constructor):
        # --- Mock GReaT setup ---
        mock_great_instance = mock_great_constructor.return_value
        mock_great_instance.fit.return_value = None
        mock_great_instance.model = MagicMock() # mock the underlying llm
        mock_great_instance.tokenizer = MagicMock() # mock the tokenizer

        # Configure mock_great_instance.impute
        # It should take a DataFrame and fill NaNs in 'Age' and 'Occupation' for this test
        def mock_impute_logic(df_to_impute, max_length):
            df_imputed = df_to_impute.copy()
            if "Age" in df_imputed.columns and pd.isna(df_imputed.loc[0, "Age"]):
                df_imputed.loc[0, "Age"] = 33 # Predictable age
            if "Occupation" in df_imputed.columns and pd.isna(df_imputed.loc[0, "Occupation"]):
                 # Based on Gender if available
                gender = df_imputed.loc[0, "Gender"]
                df_imputed.loc[0, "Occupation"] = "MockOccupationF" if gender == "F" else "MockOccupationM"
            return df_imputed
        mock_great_instance.impute.side_effect = mock_impute_logic

        # Configure mock_generate_prompt for "Name"
        # It needs to return different names based on gender and ensure novelty
        # Store original names to check against for novelty
        original_names = set(self.train_df["Name"].unique())

        # Use a dict to track generated names per gender to ensure novelty within test
        generated_names_by_gender = {"F": [], "M": []}

        def mock_prompt_logic(prompt_text, great_model_inst, device):
            # Simplified logic: just check for gender in prompt
            if "Gender: F" in prompt_text:
                candidate = "Laura"
                if candidate in original_names or candidate in generated_names_by_gender["F"]:
                    candidate = "Sophia" # Next novel female name
                generated_names_by_gender["F"].append(candidate)
                return candidate
            elif "Gender: M" in prompt_text:
                candidate = "Peter"
                if candidate in original_names or candidate in generated_names_by_gender["M"]:
                    candidate = "David"  # Next novel male name
                generated_names_by_gender["M"].append(candidate)
                return candidate
            return "UnknownName"
        mock_generate_prompt.side_effect = mock_prompt_logic

        # --- LLMGenerator setup ---
        llm_generator = LLMGenerator(
            gen_x_times=0.5, # Generate 2 new samples (0.5 * 4 original)
            text_generating_columns=["Name"],
            conditional_columns=["Gender"],
            gen_params=self.gen_params,
            # Disable post_process and adversarial for simpler focused test
            is_post_process=False
        )
        # Get the actual sampler object
        llm_sampler = llm_generator.get_object_generator()

        # --- Run generation ---
        # For this test, target can be None as LLMGenerator handles it internally if provided
        # and we are mostly concerned with feature generation.
        # test_df is also not strictly necessary if is_post_process=False
        new_train_df, _ = llm_sampler.generate_data_pipe(
            self.train_df.copy(),
            target=None, # Or self.target_df.copy() if testing with target
            test_df=self.test_df.copy(), # Or None
            only_generated_data=True # Focus on generated samples
        )

        # --- Assertions ---
        self.assertEqual(len(new_train_df), 2) # 0.5 * 4 samples

        # Check that _generate_via_prompt was called for each new sample
        self.assertEqual(mock_generate_prompt.call_count, 2)

        # Check generated names and their novelty
        for index, row in new_train_df.iterrows():
            name = row["Name"]
            gender = row["Gender"]
            self.assertNotIn(name, original_names)
            if gender == "F":
                self.assertIn(name, ["Laura", "Sophia"])
            elif gender == "M":
                self.assertIn(name, ["Peter", "David"])

            # Check imputed values
            self.assertEqual(row["Age"], 33)
            expected_occupation = "MockOccupationF" if gender == "F" else "MockOccupationM"
            self.assertEqual(row["Occupation"], expected_occupation)

        # Check gender distribution (simple check for this small sample size)
        # Ensure that the generated names align with their conditioned gender from the input.
        # The actual distribution preservation is statistical over many samples.
        # Here, we mainly check if the conditioning worked for each sample.
        generated_F = new_train_df[new_train_df["Gender"] == "F"]
        generated_M = new_train_df[new_train_df["Gender"] == "M"]

        # Depending on how attributes are sampled, we might get 1F/1M or 2F/0M or 0F/2M for 2 samples.
        # The mock_prompt_logic ensures Name matches Gender.
        # The attribute_distributions sampling in generate_data should pick F/M with 0.5 prob each.

        for _, row in generated_F.iterrows():
            self.assertIn(row["Name"], ["Laura", "Sophia"])
        for _, row in generated_M.iterrows():
            self.assertIn(row["Name"], ["Peter", "David"])

    @patch('tabgan.sampler.GReaT')
    def test_llm_generator_fallback_behavior(self, mock_great_constructor):
        # --- Mock GReaT setup for standard sampling ---
        mock_great_instance = mock_great_constructor.return_value
        mock_great_instance.fit.return_value = None

        # Expected columns for the dummy generated data by great_model_instance.sample
        # This should match self.train_df columns + self.target_df column if target is used
        # For this test, assuming target is None for simplicity in LLMGenerator call
        sample_columns = self.train_df.columns.tolist()

        # Create dummy data that model.sample() would return
        dummy_sampled_data = pd.DataFrame([
            ["SampledName1", "F", 50, "SampledOccupation1"],
            ["SampledName2", "M", 55, "SampledOccupation2"]
        ], columns=sample_columns)
        mock_great_instance.sample.return_value = dummy_sampled_data

        # --- LLMGenerator setup (no text_generating_columns) ---
        llm_generator = LLMGenerator(
            gen_x_times=0.5, # Generate 2 samples
            gen_params=self.gen_params,
            is_post_process=False
        )
        llm_sampler = llm_generator.get_object_generator()

        # --- Spy on _generate_via_prompt ---
        # We want to ensure it's NOT called in this fallback scenario
        llm_sampler._generate_via_prompt = MagicMock()

        # --- Run generation ---
        new_train_df, _ = llm_sampler.generate_data_pipe(
            self.train_df.copy(),
            target=None,
            test_df=self.test_df.copy(),
            only_generated_data=True
        )

        # --- Assertions ---
        self.assertEqual(len(new_train_df), 2)
        mock_great_instance.sample.assert_called_once()
        llm_sampler._generate_via_prompt.assert_not_called()

        # Check if the output matches the dummy_sampled_data
        pd.testing.assert_frame_equal(new_train_df.reset_index(drop=True), dummy_sampled_data.reset_index(drop=True))

# TODO: Add TestSamplerLLMWithTarget to test scenarios where target is not None for LLM.
# TODO: Add test for novelty retry logic (more complex mock for _generate_via_prompt)
# TODO: Add test for edge cases (e.g. text_generating_columns or conditional_columns are empty/None after being set)

    class TestSamplerSamplerDiffusion(TestCase):
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

    class TestSamplerSamplerDiffusion(TestCase):
        def setUp(self):
            self.train = pd.DataFrame(np.random.randint(-10, 150, size=(50, 4)), columns=list('ABCD'))
            self.target = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)), columns=list('Y'))
            self.test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
            self.gen = LLMGenerator(gen_params={"batch_size": 32, "epochs": 4, "llm": "distilgpt2",
                                                "max_length": 500})
            self.sampler = self.gen.get_object_generator()

        def test_generate_data(self):
            new_train, new_target, test_df = self.sampler.preprocess_data(self.train.copy(),
                                                                          self.target.copy(), self.test)
            gen_train, gen_target = self.sampler.generate_data(new_train, new_target, test_df)
            self.assertEqual(gen_train.shape[0], gen_target.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
            self.assertTrue(gen_train.shape[0] > new_train.shape[0])
            self.assertEqual(np.max(self.target.nunique()), np.max(new_target.nunique()))
