import torch
from torch import nn, optim
import torch.nn.functional as F


class EncoderPhotometry(nn.Module):
    def __init__(self, input_dim=6, dropout_prob=0):
        super(EncoderPhotometry, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        f = self.features(x)
        f = F.log_softmax(f, dim=1)
        return f


class MeasureZ(nn.Module):
    def __init__(self, num_gauss=10, dropout_prob=0):
        super(MeasureZ, self).__init__()

        self.ngaussians = num_gauss
        self.measure_mu = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(20, num_gauss),
        )

        self.measure_coeffs = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(20, num_gauss),
        )

        self.measure_sigma = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(20, num_gauss),
        )

    def forward(self, f):
        mu = self.measure_mu(f)
        sigma = self.measure_sigma(f)
        logmix_coeff = self.measure_coeffs(f)

        logmix_coeff = logmix_coeff - torch.logsumexp(logmix_coeff, 1)[:, None]

        return mu, sigma, logmix_coeff
