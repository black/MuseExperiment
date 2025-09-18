import numpy as np
from math import sqrt, pi, exp


class OnlineNormalEstimator:
    def __init__(self, mu_prior: float, sigma_prior: float, eta: float = 0.01):
        """
        Initialize the online estimator with a prior normal distribution.

        Parameters:
        - mu_prior: prior mean
        - sigma_prior: prior standard deviation
        - eta: learning rate (0 < eta <= 1)
        """
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.eta = eta
        self.reset()

    def update(self, x: float):
        """Add a sample and update the model parameters."""
        # Update mean with exponential moving average
        self.mu = (1 - self.eta) * self.mu + self.eta * x

        # Update variance estimate (EMA of squared deviation)
        deviation = (x - self.mu) ** 2
        self.var = (1 - self.eta) * self.var + self.eta * deviation
        self.sigma = np.sqrt(max(self.var, 1e-8))  # numerical stability

    def predict(self, x: float) -> float:
        """Return the z-score of the sample under current model."""
        return (x - self.mu) / self.sigma if self.sigma > 0 else 0.0

    def get_model_parameters(self) -> dict:
        """Return current mu and sigma as a dictionary."""
        return {"mu": self.mu, "sigma": self.sigma}

    def reset(self):
        """Reset the model back to prior parameters."""
        self.mu = self.mu_prior
        self.sigma = self.sigma_prior
        self.var = self.sigma_prior ** 2


class OnlineTwoStateGMM:
    
    def __init__(self, mu_prior, sigma_prior, pi_prior=None, eta=0.01, pi_min=0.05, min_sep=0.1):
        """
        mu_prior: list/array of length 2, prior means
        sigma_prior: list/array of length 2, prior std deviations
        pi_prior: list/array of length 2, prior mixture weights (sum to 1). If None, set to [0.5, 0.5]
        eta: learning rate for updates
        pi_min: minimum allowed mixture weight (lower bound)
        """
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
        # E-step
        p1 = self.pi[0] * self._gaussian_pdf(x, self.mu[0], self.sigma[0])
        p2 = self.pi[1] * self._gaussian_pdf(x, self.mu[1], self.sigma[1])
        total = p1 + p2
        gamma1 = p1 / total if total > 0 else 0.5
        gamma2 = 1 - gamma1

        # M-step
        self.mu[0] += self.eta * gamma1 * (x - self.mu[0])
        self.mu[1] += self.eta * gamma2 * (x - self.mu[1])

        self.sigma[0] += self.eta * gamma1 * (((x - self.mu[0]) ** 2) - self.sigma[0]**2) / (2*self.sigma[0])
        self.sigma[1] += self.eta * gamma2 * (((x - self.mu[1]) ** 2) - self.sigma[1]**2) / (2*self.sigma[1])

        self.pi[0] += self.eta * (gamma1 - self.pi[0])
        self.pi[1] = 1.0 - self.pi[0]
        self.pi = np.clip(self.pi, self.pi_min, 1 - self.pi_min)
        self.pi /= self.pi.sum()

        # ---- Mean separation safeguard ----
        dist = abs(self.mu[0] - self.mu[1])
        if dist < self.min_sep:
            mid = (self.mu[0] + self.mu[1]) / 2
            self.mu[0] = mid - self.min_sep / 2
            self.mu[1] = mid + self.min_sep / 2
            
            if self.mu[0] < 0:
                self.mu[1] += abs(self.mu[0])
                self.mu[0] += abs(self.mu[0])

    def predict(self, x):
        """Return probability of belonging to each Gaussian."""
        p1 = self.pi[0] * self._gaussian_pdf(x, self.mu[0], self.sigma[0])
        p2 = self.pi[1] * self._gaussian_pdf(x, self.mu[1], self.sigma[1])
        total = p1 + p2
        if total == 0:
            return np.array([0.5, 0.5])
        return np.array([p1 / total, p2 / total])

    def get_model_parameters(self):
        """Return current estimates of means and std deviations."""
        return {
            "mu": self.mu.copy(),
            "sigma": self.sigma.copy(),
            "pi": self.pi.copy()
        }

    def reset(self):
        """Reset to priors."""
        self.mu = self.mu_prior.copy()
        self.sigma = self.sigma_prior.copy()
        self.pi = self.pi_prior.copy()






def demo_GMM(): 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from scipy.stats import norm
    
    TEST_SAFEGUARD = False

    gmm = OnlineTwoStateGMM(mu_prior=[0, 1], sigma_prior=[0.2, 0.2], pi_prior=[0.5, 0.5], eta=0.01, pi_min=0.05)

    means = []
    sigmas = []

    mu1, mu2, s1, s2 = (0.4, 0.6, 0.1, 0.1)

    for i in range(10000):
        if TEST_SAFEGUARD:
            x = np.random.randn() * 0.1 + 0.1
        else:
            if np.random.random() < 0.5:
                x = np.random.randn() * s2 + mu2
            else:
                x = np.random.randn() * s1 + mu1

        gmm.update(x)
        params = gmm.get_model_parameters()
        means.append(params["mu"])
        sigmas.append(params["sigma"])   # <-- FIX: store sigma, not mu again

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    PALETTE = sns.color_palette("husl", 2)
    means = np.array(means)
    sigmas = np.array(sigmas)

    # Final parameters
    final_mu = means[-1]
    final_sigma = sigmas[-1]

    # Grid for the side distributions
    x_grid = np.linspace(min(final_mu)-3*max(final_sigma),
                         max(final_mu)+3*max(final_sigma), 500)

    # --- set up figure with gridspec ---
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1], wspace=0.05)

    # --- Main time-series axis ---
    ax0 = plt.subplot(gs[0])

    ax0.plot(means[:,0], color=PALETTE[0], label="Component 1")
    ax0.fill_between(np.arange(len(means)),
                     means[:,0] - sigmas[:,0],
                     means[:,0] + sigmas[:,0],
                     color=PALETTE[0], alpha=0.2)

    ax0.plot(means[:,1], color=PALETTE[1], label="Component 2")
    ax0.fill_between(np.arange(len(means)),
                     means[:,1] - sigmas[:,1],
                     means[:,1] + sigmas[:,1],
                     color=PALETTE[1], alpha=0.2)

    ax0.set_xlabel('Iteration')
    ax0.set_ylabel('GMM means ± sigma')
    ax0.set_title('GMM tracking with final distributions on right')
    ax0.legend(loc='center right')

    # --- Side distribution axis ---
    ax1 = plt.subplot(gs[1], sharey=ax0)

    for k, color in enumerate(PALETTE):
        pdf = norm.pdf(x_grid, loc=final_mu[k], scale=final_sigma[k])
        ax1.plot(pdf, x_grid, color=color, lw=2)
        ax1.fill_betweenx(x_grid, 0, pdf, color=color, alpha=0.2)

    ax1.set_xlabel("Density")
    ax1.yaxis.set_visible(False)   # hide y-axis labels
    sns.despine(ax=ax1, left=True)

    plt.show()
    


def demo_zScore(): 
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from scipy.stats import norm

    model = OnlineNormalEstimator(mu_prior=0.5, sigma_prior=0.2, eta=0.01)

    means = []
    sigmas = []

    # True data parameters
    mu1, mu2, s1, s2 = (0.4, 0.6, 0.1, 0.1)

    # Generate data and update model
    for i in range(10000):
        if np.random.random() < 0.5:
            x = np.random.randn() * s2 + mu2
        else:
            x = np.random.randn() * s1 + mu1

        model.update(x)
        params = model.get_model_parameters()
        means.append(params["mu"])
        sigmas.append(params["sigma"])

    # --- Plotting ---
    sns.set_style("whitegrid")
    PALETTE = sns.color_palette("husl", 1)

    means = np.array(means)
    sigmas = np.array(sigmas)

    final_mu = means[-1]
    final_sigma = sigmas[-1]

    x_grid = np.linspace(final_mu - 3*final_sigma,
                         final_mu + 3*final_sigma, 500)

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4,1], wspace=0.05)

    # --- Time series axis ---
    ax0 = plt.subplot(gs[0])
    ax0.plot(means, color="black", label="Mean")
    ax0.fill_between(np.arange(len(means)),
                     means - sigmas,
                     means + sigmas,
                     color="black", alpha=0.2)

    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Mean ± Sigma")
    ax0.set_title("Online Normal Model tracking")
    ax0.legend(loc="center right")

    # --- Side distribution ---
    ax1 = plt.subplot(gs[1], sharey=ax0)

    pdf = norm.pdf(x_grid, loc=final_mu, scale=final_sigma)
    ax1.plot(pdf, x_grid, color="black", lw=2)
    ax1.fill_betweenx(x_grid, 0, pdf, color="black", alpha=0.2)

    ax1.set_xlabel("Density")
    ax1.yaxis.set_visible(False)
    sns.despine(ax=ax1, left=True)

    plt.show()

if __name__ == "__main__":
    #demo_GMM()
    demo_zScore()