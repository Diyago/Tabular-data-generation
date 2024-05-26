from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
import pandas as pd
import numpy as np

# random input data
train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# generate data
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test, )
new_train2, new_target2 = GANGenerator(gen_params={"batch_size": 500, "epochs": 10, "patience": 5}).generate_data_pipe(
    train, target, test, )
new_train3, new_target3 = ForestDiffusionGenerator().generate_data_pipe(train, target, test, )
new_train4, new_target4 = LLMGenerator(gen_params={"batch_size": 32,
                                                   "epochs": 4, "llm": "distilgpt2",
                                                   "max_length": 500}).generate_data_pipe(train, target, test, )