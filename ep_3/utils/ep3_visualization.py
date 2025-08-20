import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from typing import Sequence

# Default color palette
PALETTE = sns.color_palette("husl", 2)


def plot_power_spectra(freqs_open: np.ndarray,
                       power_open: np.ndarray,
                       freqs_closed: np.ndarray,
                       power_closed: np.ndarray,
                       palette=PALETTE) -> None:
    """Plot average power spectra for eyes open vs eyes closed."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    plt.axvspan(8, 13, color='gray', alpha=0.2, label='Alpha Band (8–13 Hz)')
    plt.plot(freqs_open, power_open, label='Eyes Open', color=palette[0])
    plt.plot(freqs_closed, power_closed, label='Eyes Closed', color=palette[1])
    
    plt.xlim([0, 50])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Average Power Spectrum ((TP9 + TP10)/2)')
    plt.legend()
    plt.show()


def plot_alpha_distribution(rel_alpha_open: np.ndarray,
                            rel_alpha_closed: np.ndarray,
                            mean_open: float,
                            mean_closed: float,
                            auc_value: float,
                            fpr: Sequence[float],
                            tpr: Sequence[float],
                            palette=PALETTE) -> None:
    """Plot alpha power distribution with ROC curve."""
    plt.figure(figsize=(10, 4))
    
    # Histogram
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, 1, 31)
    plt.hist(rel_alpha_open, bins=bins, alpha=0.6, label='Eyes Open', color=palette[0])
    plt.hist(rel_alpha_closed, bins=bins, alpha=0.6, label='Eyes Closed', color=palette[1])
    plt.axvline(mean_open, color=palette[0], linestyle='--', label=f'Open Mean: {mean_open:.2f}')
    plt.axvline(mean_closed, color=palette[1], linestyle='--', label=f'Closed Mean: {mean_closed:.2f}')
    plt.legend()
    
    # ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.show()


def plot_std_explanation(rel_alpha_open: np.ndarray, palette=["#1f77b4", "#ff7f0e"]) -> None:
    """Illustrate standard deviation and normal fit for eyes open alpha power."""
    plt.figure(figsize=(10, 6))

    # Step 1: Standard Normal Distribution
    plt.subplot(2, 1, 1)
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)
    plt.plot(x, y, color=palette[0], label="Standard Normal (μ=0, σ=1)")
    plt.axvline(0, color="k", linestyle="--", label="Mean (0)")
    plt.fill_between(x, y, 0, where=(x > -1) & (x < 1), color=palette[0], alpha=0.3, label="±1σ")
    plt.title("Step 1: Standard Normal Distribution")
    plt.legend()

    # Step 2: Fit normal to data
    plt.subplot(2, 1, 2)
    mean_open = np.mean(rel_alpha_open)
    std_open = np.std(rel_alpha_open)
    x_fit = np.linspace(min(rel_alpha_open), max(rel_alpha_open), 500)
    y_fit = norm.pdf(x_fit, loc=mean_open, scale=std_open)
    
    plt.hist(rel_alpha_open, bins=np.linspace(0, 1, 31), density=True, alpha=0.6,
             color=PALETTE[0], label="Eyes Open Data")
    plt.plot(x_fit, y_fit, color="k", linewidth=2, label=f"Fit Normal (μ={mean_open:.2f}, σ={std_open:.2f})")
    plt.axvline(mean_open, color=PALETTE[0], linestyle="--", label="Mean")
    plt.fill_between(x_fit, y_fit, 0, where=(x_fit > mean_open-std_open) & (x_fit < mean_open+std_open),
                     color=PALETTE[0], alpha=0.3, label="±1σ range")
    plt.title("Step 2: Normal Fit on Eyes Open Data")
    plt.legend()
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.show()


def plot_std_zscore_explanation(rel_alpha_open: np.ndarray, rel_alpha_closed: np.ndarray,
                                palette=["#1f77b4", "#ff7f0e"]) -> None:
    """Illustrate z-score standardization across eyes open and closed."""
    plt.figure(figsize=(10, 6))

    # Step 1: Standard Normal
    plt.subplot(2, 1, 1)
    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)
    plt.plot(x, y, color=palette[0], label="Standard Normal (μ=0, σ=1)")
    plt.axvline(0, color="k", linestyle="--", label="Mean (0)")
    plt.fill_between(x, y, 0, where=(x > -1) & (x < 1), color=palette[0], alpha=0.3, label="±1σ")
    plt.title("Step 1: Standard Normal Distribution")
    plt.legend()

    # Step 2: Fit normal to combined data
    plt.subplot(2, 1, 2)
    combined = np.concatenate((rel_alpha_open, rel_alpha_closed))
    mean_comb = np.mean(combined)
    std_comb = np.std(combined)
    x_fit = np.linspace(min(combined), max(combined), 500)
    y_fit = norm.pdf(x_fit, loc=mean_comb, scale=std_comb)
    
    plt.hist(rel_alpha_open, bins=np.linspace(0, 1, 31), density=True, alpha=0.6, color=PALETTE[0], label="Eyes Open")
    plt.hist(rel_alpha_closed, bins=np.linspace(0, 1, 31), density=True, alpha=0.6, color=PALETTE[1], label="Eyes Closed")
    plt.plot(x_fit, y_fit, color="k", linewidth=2, label=f"Fit Normal (μ={mean_comb:.2f}, σ={std_comb:.2f})")
    plt.axvline(mean_comb, color=PALETTE[0], linestyle="--", label="Mean")
    plt.fill_between(x_fit, y_fit, 0, where=(x_fit > mean_comb-std_comb) & (x_fit < mean_comb+std_comb),
                     color=PALETTE[0], alpha=0.3, label="±1σ range")
    plt.title("Step 2: Normal Fit on Combined Data")
    plt.legend()
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.show()


def _gaussian_boundary_equal_priors(mu0: float, s0: float, mu1: float, s1: float) -> float:
    """Compute decision boundary where two Gaussians are equal (equal priors)."""
    a = 1.0/s1**2 - 1.0/s0**2
    b = -2.0*mu1/s1**2 + 2.0*mu0/s0**2
    c = (mu1**2)/s1**2 - (mu0**2)/s0**2 - 2.0*np.log(s1/s0)

    if np.isclose(a, 0.0):
        return (mu0 + mu1) / 2.0

    disc = max(b*b - 4*a*c, 0.0)
    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)

    lo, hi = sorted([mu0, mu1])
    candidates = [r for r in (r1, r2) if lo <= r <= hi]
    if candidates:
        return candidates[0]
    mid = 0.5*(mu0 + mu1)
    return r1 if abs(r1 - mid) < abs(r2 - mid) else r2


def plot_two_class_likelihood_demo(palette=PALETTE) -> None:
    """Demonstrate two-class Gaussian likelihoods with decision boundary."""
    sns.set_style("whitegrid")
    np.random.seed(42)

    data0 = np.random.normal(0.2, 0.2, 500)
    data1 = np.random.normal(0.5, 0.2, 500)
    mu0, s0 = np.mean(data0), np.std(data0)
    mu1, s1 = np.mean(data1), np.std(data1)
    boundary_x = _gaussian_boundary_equal_priors(mu0, s0, mu1, s1)

    x = np.linspace(min(mu0-4*s0, mu1-4*s1), max(mu0+4*s0, mu1+4*s1), 1000)
    pdf0 = norm.pdf(x, mu0, s0)
    pdf1 = norm.pdf(x, mu1, s1)

    plt.figure(figsize=(8, 5))
    plt.hist(data0, bins=30, density=True, alpha=0.3, color=palette[0], label="Class 0 Data")
    plt.hist(data1, bins=30, density=True, alpha=0.3, color=palette[1], label="Class 1 Data")
    plt.plot(x, pdf0, color=palette[0], lw=2, label=f"Class 0 Fit (μ={mu0:.2f}, σ={s0:.2f})")
    plt.plot(x, pdf1, color=palette[1], lw=2, label=f"Class 1 Fit (μ={mu1:.2f}, σ={s1:.2f})")
    plt.axvline(boundary_x, color="k", linestyle="--", lw=2, label=f"Decision Boundary: {boundary_x:.3f}")
    plt.title("Two-Class Gaussian Likelihoods (Equal Priors)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_single_gaussian_demo(palette=PALETTE) -> None:
    """Demonstrate single Gaussian fit across two classes."""
    sns.set_style("whitegrid")
    np.random.seed(42)

    data0 = np.random.normal(0.2, 0.2, 500)
    data1 = np.random.normal(0.5, 0.2, 500)
    data_all = np.concatenate([data0, data1])
    mu, sigma = np.mean(data_all), np.std(data_all)
    boundary_x = mu

    x = np.linspace(min(data_all)-0.2, max(data_all)+0.2, 1000)
    pdf_all = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(8, 5))
    plt.hist(data0, bins=30, density=True, alpha=0.3, color=palette[0], label="Class 0 Data")
    plt.hist(data1, bins=30, density=True, alpha=0.3, color=palette[1], label="Class 1 Data")

    plt.plot(x, pdf_all, color="k", lw=2, label=f"Single Gaussian Fit (μ={mu:.2f}, σ={sigma:.2f})")

    plt.axvline(boundary_x, color="r", linestyle="--", lw=2, label=f"Decision Boundary (mean = {boundary_x:.2f})")

    plt.title("Single Gaussian Fit with Decision Boundary at the Mean")
    plt.legend()
    plt.tight_layout()
    plt.show()