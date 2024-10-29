import torch
from torch import nn
import torch.nn.functional as F


class EncoderPhotometry(nn.Module):
    """Encoder for photometric data.

    This neural network encodes photometric features into a lower-dimensional representation.

    Attributes:
        features (nn.Sequential): A sequential container of layers used for encoding.
    """

    def __init__(self, input_dim: int = 6, dropout_prob: float = 0) -> None:
        """Initializes the EncoderPhotometry module.

        Args:
            input_dim (int): Number of input features (default is 6).
            dropout_prob (float): Probability of dropout (default is 0).
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Log softmax output of shape (batch_size, 10).
        """
        f = self.features(x)
        f = F.log_softmax(f, dim=1)
        return f


class MeasureZ(nn.Module):
    """Model to measure redshift parameters.

    This model estimates the parameters of a mixture of Gaussians used for measuring redshift.

    Attributes:
        ngaussians (int): Number of Gaussian components in the mixture.
        measure_mu (nn.Sequential): Sequential model to measure the mean (mu).
        measure_coeffs (nn.Sequential): Sequential model to measure the mixing coefficients.
        measure_sigma (nn.Sequential): Sequential model to measure the standard deviation (sigma).
    """

    def __init__(self, num_gauss: int = 10, dropout_prob: float = 0) -> None:
        """Initializes the MeasureZ module.

        Args:
            num_gauss (int): Number of Gaussian components (default is 10).
            dropout_prob (float): Probability of dropout (default is 0).
        """
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

    def forward(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to measure redshift parameters.

        Args:
            f (torch.Tensor): Input tensor of shape (batch_size, 10).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - mu (torch.Tensor): Mean parameters of shape (batch_size, num_gauss).
                - sigma (torch.Tensor): Standard deviation parameters of shape (batch_size, num_gauss).
                - logmix_coeff (torch.Tensor): Log mixing coefficients of shape (batch_size, num_gauss).
        """
        mu = self.measure_mu(f)
        sigma = self.measure_sigma(f)
        logmix_coeff = self.measure_coeffs(f)

        # Normalize logmix_coeff to get valid mixture coefficients
        logmix_coeff = logmix_coeff - torch.logsumexp(logmix_coeff, dim=1, keepdim=True)

        return mu, sigma, logmix_coeff
