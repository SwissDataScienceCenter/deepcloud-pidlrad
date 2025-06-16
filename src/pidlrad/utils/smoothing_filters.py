import torch


def gassuian_kernel(sigmas, size=71):
    gk = torch.zeros(size, size)
    x = torch.arange(0, size)
    means = torch.arange(0, size)

    for i, (mean, sigma) in enumerate(zip(means, sigmas)):
        g = torch.exp(-0.5 * ((x - mean) / max(sigma, 1e-10)) ** 2)
        gk[i, :] = g / g.sum()
    return torch.flip(gk.T, [0, 1])


def exponential_sigma(size=71, min_=1e-1, max_=5):
    return torch.exp(
        torch.linspace(
            torch.log(torch.tensor(min_)), torch.log(torch.tensor(max_)), size
        )
    )
