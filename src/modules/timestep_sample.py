import torch
from torch.distributions import Normal

def truncated_normal(size, mean=0., std=1., low=0., high=999.):

    normal_dist = Normal(mean, std)

    # Calculate the CDF values for the bounds
    low_cdf = normal_dist.cdf(torch.tensor(low))
    high_cdf = normal_dist.cdf(torch.tensor(high))

    # Sample uniformly from the truncated CDF range
    uniform_samples = torch.rand(size) * (high_cdf - low_cdf) + low_cdf

    # Invert the CDF to get the truncated normal samples
    truncated_samples = normal_dist.icdf(uniform_samples)

    return truncated_samples.round().long()