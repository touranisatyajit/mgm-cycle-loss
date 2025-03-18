
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
import torch
#https://github.com/src-d/lapjv
from lapjv import lapjv
from lap import lapjv as lapjv_unbalanced #supports unbalanced problems
import matplotlib.pyplot as plt

def to_var(x):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        x = x.cuda()
    return x


def sample_fn(shape):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability

    Returns:
    A sample of standard Gumbel random variables
    """
    # Sample from Gumbel with expectancy 0 and variance
    beta = np.sqrt(0.2/(np.square(np.pi)))

    # Sample from Gumbel with expectancy 0 and variance 2
    #beta = np.sqrt(12./(np.square(np.pi)))

    # Sample from Gumbel with expectancy 0 and variance 3
    #beta = np.sqrt(18./(np.square(np.pi)))

    mu = -beta*np.euler_gamma

    U = np.random.gumbel(loc=mu, scale=beta, size=shape)
    return torch.from_numpy(U).float()

def perturb_scoreMatrix_unbalanced(score, samples_per_num, noise_factor, sigma):
    """
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, M])
      or 3D tensor (a batch of matrices of shape = [batch_size, N, M])

    Returns:
    log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, M] of
          noisy samples of log_alpha, If n_samples = 1 then the output is 3D.
    """
    n = score.size()[1]
    m = score.size()[2]
    score = score.view(-1, n, m)
    batch_size = score.size()[0]

    if samples_per_num > 1:
        score_tiled = score.repeat(samples_per_num, 1, 1)
    else:
        score_tiled = score

    noise_sigma_tiled = to_var(torch.zeros((batch_size * samples_per_num, n, m)))
    if noise_factor == True:
        noise = to_var(sample_fn([batch_size * samples_per_num, n, m]))

        # rescale noise according to sigma
        sigma_tiled = sigma.repeat(samples_per_num, 1)
        for bm in range(batch_size * samples_per_num):
            noise_sigma_tiled[bm] = sigma_tiled[bm] * noise[bm]

        perturbed_score = score_tiled + noise_sigma_tiled

    else:
        perturbed_score = score_tiled

    return perturbed_score, noise_sigma_tiled


