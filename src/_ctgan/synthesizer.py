import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional
from tqdm.autonotebook import tqdm

from _ctgan.conditional import ConditionalGenerator
from _ctgan.models import Discriminator, Generator
from _ctgan.sampler import Sampler
from _ctgan.transformer import DataTransformer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.verbose = verbose

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stoping for GAN. Best score: {:.2f} with patience = {}".format(self.best_score,
                                                                                               self.patience))
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class _CTGANSynthesizer:
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Wheight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
    """

    def __init__(
        self,
        embedding_dim=128,
        gen_dim=(256, 256),
        dis_dim=(256, 256),
        l2scale=1e-6,
        batch_size=500,
        patience=25,
    ):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.patience = patience
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == "tanh":
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == "softmax":
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                raise AssertionError

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == "tanh":
                st += item[0]
                skip = True

            elif item[1] == "softmax":
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction="none",
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                raise AssertionError

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def fit(self, train_data, discrete_columns=(), epochs=300, log_frequency=True):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data, self.transformer.output_info, log_frequency
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim
        ).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt, self.dis_dim
        ).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.9),
            weight_decay=self.l2scale,
        )
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        if self.batch_size % 2 != 0:
            raise ValueError("batch_size should even, but {} is provided".format(self.batch_size))
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        train_losses = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=False)

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)

        for i in tqdm(range(epochs), desc="Training CTGAN, epochs:"):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype("float32")).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                pen = discriminator.calc_gradient_penalty(
                    real_cat, fake_cat, self.device
                )
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                train_losses.append(loss_d.item())
                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                train_losses.append(loss_g.item())
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
            early_stopping(np.average(train_losses))
            if early_stopping.early_stop:
                logging.info("Early stopping in GAN training!")
                break
            train_losses = []

    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)
