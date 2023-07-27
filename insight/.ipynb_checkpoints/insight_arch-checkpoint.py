from torch import nn, optim
import torch
class Photoz_network(nn.Module):
    def __init__(self, num_gauss=10, dropout_prob=0):
        super(Photoz_network, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6, 10),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(10, 30),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(50, 70),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(70, 100)
        )
        
        self.measure_mu = nn.Sequential(
            nn.Linear(100, 80),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(80, 70),
            nn.Dropout(dropout_prob),
            nn.ReLU(),        
            nn.Linear(70, 60),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(60, 50),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(50, num_gauss)
        )

        self.measure_coeffs = nn.Sequential(
            nn.Linear(100, 80),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(80, 70),
            nn.Dropout(dropout_prob),
            nn.ReLU(),        
            nn.Linear(70, 60),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(60, 50),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(50, num_gauss)
        )

        self.measure_sigma = nn.Sequential(
            nn.Linear(100, 80),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(80, 70),
            nn.Dropout(dropout_prob),
            nn.ReLU(),        
            nn.Linear(70, 60),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(60, 50),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(50, num_gauss)
        )
        
    def forward(self, x):
        f = self.features(x)
        mu = self.measure_mu(f)
        sigma = self.measure_sigma(f)
        logmix_coeff = self.measure_coeffs(f)
        
        logmix_coeff = logmix_coeff - torch.logsumexp(logmix_coeff, 1)[:,None]

        return mu, sigma, logmix_coeff
    

