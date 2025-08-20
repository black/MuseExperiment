import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import norm
from math import sqrt, pi

# -----------------------------
# Online Normal Estimator
# -----------------------------
class OnlineNormalEstimator:
    def __init__(self, mu_prior: float, sigma_prior: float, eta: float = 0.01):
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.eta = eta
        self.reset()

    def update(self, x: float):
        self.mu = (1 - self.eta) * self.mu + self.eta * x
        deviation = (x - self.mu) ** 2
        self.var = (1 - self.eta) * self.var + self.eta * deviation
        self.sigma = np.sqrt(max(self.var, 1e-8))

    def predict(self, x: float) -> float:
        return (x - self.mu) / self.sigma if self.sigma > 0 else 0.0

    def get_model_parameters(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma}

    def reset(self):
        self.mu = self.mu_prior
        self.sigma = self.sigma_prior
        self.var = self.sigma_prior ** 2


# -----------------------------
# Online Two-State GMM
# -----------------------------
class OnlineTwoStateGMM:
    def __init__(self, mu_prior, sigma_prior, pi_prior=None, eta=0.01, pi_min=0.05, min_sep=0.1):
        self.mu_prior = np.array(mu_prior, dtype=float)
        self.sigma_prior = np.array(sigma_prior, dtype=float)
        self.pi_prior = np.array(pi_prior if pi_prior is not None else [0.5, 0.5], dtype=float)
        self.eta = eta
        self.pi_min = pi_min
        self.min_sep = min_sep
        self.reset()

    def _gaussian_pdf(self, x, mu, sigma):
        coeff = 1.0 / (sqrt(2 * pi) * sigma)
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return coeff * np.exp(exponent)

    def update(self, x):
        p1 = self.pi[0] * self._gaussian_pdf(x, self.mu[0], self.sigma[0])
        p2 = self.pi[1] * self._gaussian_pdf(x, self.mu[1], self.sigma[1])
        total = p1 + p2
        gamma1 = p1 / total if total > 0 else 0.5
        gamma2 = 1 - gamma1

        self.mu[0] += self.eta * gamma1 * (x - self.mu[0])
        self.mu[1] += self.eta * gamma2 * (x - self.mu[1])

        self.sigma[0] += self.eta * gamma1 * (((x - self.mu[0]) ** 2 - self.sigma[0] ** 2) / (2*self.sigma[0]))
        self.sigma[1] += self.eta * gamma2 * (((x - self.mu[1]) ** 2 - self.sigma[1] ** 2) / (2*self.sigma[1]))

        self.pi[0] += self.eta * (gamma1 - self.pi[0])
        self.pi[1] = 1.0 - self.pi[0]
        self.pi = np.clip(self.pi, self.pi_min, 1 - self.pi_min)
        self.pi /= self.pi.sum()

        # Safeguard for minimum mean separation
        dist = abs(self.mu[0] - self.mu[1])
        if dist < self.min_sep:
            mid = (self.mu[0] + self.mu[1]) / 2
            self.mu[0] = mid - self.min_sep / 2
            self.mu[1] = mid + self.min_sep / 2

    def predict(self, x):
        p1 = self.pi[0] * self._gaussian_pdf(x, self.mu[0], self.sigma[0])
        p2 = self.pi[1] * self._gaussian_pdf(x, self.mu[1], self.sigma[1])
        total = p1 + p2
        return np.array([0.5, 0.5]) if total == 0 else np.array([p1 / total, p2 / total])

    def get_model_parameters(self):
        return {"mu": self.mu.copy(), "sigma": self.sigma.copy(), "pi": self.pi.copy()}

    def reset(self):
        self.mu = self.mu_prior.copy()
        self.sigma = self.sigma_prior.copy()
        self.pi = self.pi_prior.copy()


# -----------------------------
# Demo Functions
# -----------------------------
def demo_GMM():
    gmm = OnlineTwoStateGMM(mu_prior=[0, 1], sigma_prior=[0.2, 0.2], eta=0.01)
    means, sigmas = [], []

    mu1, mu2, s1, s2 = 0.4, 0.6, 0.1, 0.1
    for _ in range(10000):
        x = np.random.randn() * s2 + mu2 if np.random.random() < 0.5 else np.random.randn() * s1 + mu1
        gmm.update(x)
        params = gmm.get_model_parameters()
        means.append(params["mu"])
        sigmas.append(params["sigma"])

    means = np.array(means)
    sigmas = np.array(sigmas)
    final_mu, final_sigma = means[-1], sigmas[-1]

    x_grid = np.linspace(min(final_mu)-3*max(final_sigma), max(final_mu)+3*max(final_sigma), 500)

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1], wspace=0.05)
    ax0 = plt.subplot(gs[0])
    PALETTE = sns.color_palette("husl", 2)

    for i in [0, 1]:
        ax0.plot(means[:,i], color=PALETTE[i], label=f"Component {i+1}")
        ax0.fill_between(np.arange(len(means)), means[:,i]-sigmas[:,i], means[:,i]+sigmas[:,i], color=PALETTE[i], alpha=0.2)

    ax0.set_xlabel('Iteration')
    ax0.set_ylabel('GMM means ± sigma')
    ax0.set_title('GMM tracking with final distributions on right')
    ax0.legend(loc='center right')

    ax1 = plt.subplot(gs[1], sharey=ax0)
    for k, color in enumerate(PALETTE):
        pdf = norm.pdf(x_grid, loc=final_mu[k], scale=final_sigma[k])
        ax1.plot(pdf, x_grid, color=color, lw=2)
        ax1.fill_betweenx(x_grid, 0, pdf, color=color, alpha=0.2)
    ax1.set_xlabel("Density")
    ax1.yaxis.set_visible(False)
    sns.despine(ax=ax1, left=True)
    plt.show()


def demo_zScore():
    model = OnlineNormalEstimator(mu_prior=0.5, sigma_prior=0.2, eta=0.01)
    means, sigmas = [], []

    mu1, mu2, s1, s2 = 0.4, 0.6, 0.1, 0.1
    for _ in range(10000):
        x = np.random.randn() * s2 + mu2 if np.random.random() < 0.5 else np.random.randn() * s1 + mu1
        model.update(x)
        params = model.get_model_parameters()
        means.append(params["mu"])
        sigmas.append(params["sigma"])

    means = np.array(means)
    sigmas = np.array(sigmas)
    final_mu, final_sigma = means[-1], sigmas[-1]

    x_grid = np.linspace(final_mu-3*final_sigma, final_mu+3*final_sigma, 500)

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1], wspace=0.05)
    ax0 = plt.subplot(gs[0])

    ax0.plot(means, color="black", label="Mean")
    ax0.fill_between(np.arange(len(means)), means-sigmas, means+sigmas, color="black", alpha=0.2)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Mean ± Sigma")
    ax0.set_title("Online Normal Model tracking")
    ax0.legend(loc="center right")

    ax1 = plt.subplot(gs[1], sharey=ax0)
    pdf = norm.pdf(x_grid, loc=final_mu, scale=final_sigma)
    ax1.plot(pdf, x_grid, color="black", lw=2)
    ax1.fill_betweenx(x_grid, 0, pdf, color="black", alpha=0.2)
    ax1.set_xlabel("Density")
    ax1.yaxis.set_visible(False)
    sns.despine(ax=ax1, left=True)
    plt.show()
