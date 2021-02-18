[![CodeFactor](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data/badge)](https://www.codefactor.io/repository/github/diyago/gan-for-tabular-data)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# GANs for tabular  data
<img src="https://raw.githubusercontent.com/Diyago/GAN-for-tabular-data/e5a4d437655261755de962b9779c73203611d921/images/logo%20tabular%20gan.svg" height="15%" width="15%">
We well know GANs for success in the realistic image generation. However, they can be applied in tabular data generation. We will review and examine some recent papers about tabular GANs in action.

* Arxiv article: ["Tabular GANs for uneven distribution"](https://arxiv.org/abs/2010.00638)
* Medium post: [GANs for tabular data](https://towardsdatascience.com/review-of-gans-for-tabular-data-a30a2199342)

**Library goal**

Let say we have **T_train** and **T_test** (train and test set respectively). 
We need to train the model on **T_train** and make predictions on **T_test**. 
However, we will increase the train by generating new data by GAN or by sampling
train, somehow similar to **T_test**, without using ground truth labels.

## Acknowledgments

The author would like to thank Open Data Science community [7] for many
valuable discussions and educational help in the growing field of machine and
deep learning. Also, special big thanks to Sber [8] for allowing solving
such tasks and providing computational resources.

## References

[1] Jonathan Hui. GAN — What is Generative Adversarial Networks GAN? (2018), medium article

[2]Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. Generative Adversarial Networks (2014). arXiv:1406.2661

[3] Lei Xu LIDS, Kalyan Veeramachaneni. Synthesizing Tabular Data using Generative Adversarial Networks (2018). arXiv:1811.11264v1 [cs.LG]

[4] Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni. Modeling Tabular Data using Conditional GAN (2019). arXiv:1907.00503v2 [cs.LG]

[5] Denis Vorotyntsev. Benchmarking Categorical Encoders (2019). Medium post

[6] Insaf Ashrapov. GAN-for-tabular-data (2020). Github repository.

[7] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. Analyzing and Improving the Image Quality of StyleGAN (2019) arXiv:1912.04958v2 [cs.CV]

[8]  ODS.ai: Open data science (2020), https://ods.ai/

[9]  Sber (2020), https://www.sberbank.ru/
