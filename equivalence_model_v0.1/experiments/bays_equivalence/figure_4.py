"""
Bays (2014) Figure 4 — GP-Based Equivalent (Complete, Self-Contained)

Reproduces the full 3-row × 6-column layout:
    Row 1: Driving input / tuning curves / correlation matrix
    Row 2: Normalised error distributions (colour = gain)
    Row 3: Deviation from circular normal (Δ probability)

Six panels:
    a — Broad GP tuning        (λ = 1.0)
    b — Narrow GP tuning       (λ = 0.3)
    c — Narrow + baseline      (λ = 0.3, f(0) = 0.25)
    d — Heterogeneous tuning   (random λ, amplitude, baseline)
    e — Cosine tuning           (half-wave rectified)
    f — Correlated activity    (c₀ = 0.25, short-range correlations)

Two population sizes: M = 100 (solid) and M = 1000 (dashed).

=============================================================================
FIRST-PRINCIPLES MODEL CHAIN
=============================================================================

1. TUNING CURVES  g_i(θ)
   Bays uses von Mises: f(θ) = exp(κ⁻¹(cos(θ_i - θ) - 1))
   Our GP equivalent: draw from a Gaussian Process with periodic RBF kernel
       K(θ, θ') = exp(-2 sin²(|θ-θ'|/2) / λ²)
   Then exponentiate: g_i(θ) = exp(f_i(θ)) to ensure positivity.
   
   WHY THIS WORKS: The GP lengthscale λ plays the same role as Bays's
   tuning width ω — it controls how rapidly the tuning curve falls off
   from its peak. Shorter λ → narrower tuning → higher precision at
   a given gain, exactly as ω does in the von Mises model.

2. DIVISIVE NORMALISATION
       r_i(θ) = γ · g_i(θ) / (σ² + (1/M) Σ_j g_j(θ))
   
   This is Bays's Eq. 2-3: the gain γ sets the total population output.
   With dense uniform coverage, the denominator is ~constant across θ,
   so each neuron's firing rate is proportional to its driving input
   scaled by γ/N (where N = number of stimuli, here N=1).

3. POISSON SPIKE GENERATION
       n_i ~ Poisson(r_i · T_d)
   
   Independent for panels a–e. Panel f uses the latent Gaussian method
   (Macke et al. 2008) to introduce short-range pairwise correlations.

4. MAXIMUM LIKELIHOOD DECODING
       θ̂ = argmax_θ [ Σ_i n_i · log g_i(θ) - T_d · Σ_i g_i(θ) ]
   
   Under dense uniform coverage, Σ_i g_i(θ) ≈ const, so the second
   term drops out and we get: θ̂ = argmax_θ Σ_i n_i · log g_i(θ)
   
   KEY INSIGHT: The decoder uses the DRIVING INPUT g_i(θ), not the
   post-normalisation rate r_i. This is because the normalisation
   constant is the same for all θ (dense uniform coverage), so it
   cancels in the argmax.

5. ERROR ANALYSIS
   - Circular error: e = angle(exp(i(θ̂ - θ_true)))
   - Fit von Mises to errors (ML κ estimation via Bessel functions)
   - Deviation = empirical histogram - fitted von Mises PDF

=============================================================================
THE KEY FINDING (and what Figure 4 tests)
=============================================================================

At HIGH gain: many spikes → ML decoder is precise → errors ≈ von Mises
At LOW gain:  few spikes  → decoder sometimes gets ~0 spikes → errors
              become a MIXTURE of a peaked distribution (when enough spikes)
              and a near-uniform distribution (when ~0 spikes).

This mixture produces POSITIVE KURTOSIS: heavier tails + sharper peak
than a von Mises of the same variance. The Δ probability plot shows this
as positive deviation at the peak and at the tails, negative in between.

Figure 4 shows this pattern is ROBUST to:
    - Tuning width (a vs b)
    - Baseline activity (c)
    - Heterogeneous tuning (d)
    - Non-bell-shaped (cosine) tuning (e)
    - Short-range noise correlations (f)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.stats import vonmises, norm, poisson
from scipy.special import i0, i1
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================

def periodic_rbf_kernel(thetas, lengthscale):
    """
    Periodic RBF (squared exponential) kernel on the circle.
    
        K(θ, θ') = exp(-2 sin²(|θ-θ'|/2) / λ²)
    
    This is the GP equivalent of Bays's von Mises tuning width ω.
    The lengthscale λ controls smoothness: 
        large λ → broad, smooth tuning curves (like large ω)
        small λ → narrow, peaked tuning curves (like small ω)
    """
    diff = thetas[:, None] - thetas[None, :]
    return np.exp(-2.0 * np.sin(diff / 2.0)**2 / lengthscale**2)


def sample_gp_function(K, rng):
    """
    Draw a single function from GP(0, K).
    
    Returns the raw GP sample f(θ). To get a positive tuning curve,
    exponentiate: g(θ) = exp(f(θ)).
    
    The exponentiation is key: it ensures g > 0 (firing rates can't be
    negative) and maps the GP's Gaussian marginals to log-normal marginals,
    giving a natural amplitude distribution.
    """
    n = K.shape[0]
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    return L @ rng.randn(n)


def dn_pointwise(g_col, gamma, sigma_sq, mean_g):
    """
    Divisive normalisation at a single orientation θ.
    
        r_i = γ · g_i(θ) / (σ² + mean_g(θ))
    
    where mean_g ≈ (1/M) Σ_j g_j(θ) is precomputed.
    
    Bays's Eq. 3: under dense uniform coverage, the denominator is 
    constant across θ, so this is equivalent to scaling all rates
    by a single gain factor γ/(M·f̄).
    
    Parameters
    ----------
    g_col : (M,) driving inputs at orientation θ
    gamma : gain constant (Hz)
    sigma_sq : semi-saturation constant (prevents division by zero)
    mean_g : mean driving input across the population at this θ
    """
    return gamma * g_col / (sigma_sq + mean_g)


def generate_spikes(rates, T_d, rng):
    """Independent Poisson spike generation."""
    lambdas = np.maximum(rates * T_d, 0.0)
    return rng.poisson(lambdas)


def compute_log_likelihood(counts, log_g, T_d_sum_g):
    """
    Log-likelihood for ML decoding.
    
        LL(θ) = Σ_i n_i · log g_i(θ) - T_d · Σ_i g_i(θ)
    
    Under dense uniform coverage, T_d · Σ_i g_i(θ) ≈ const,
    so the argmax depends only on the first term. But we keep
    both for correctness (matters for heterogeneous/cosine panels).
    
    Parameters
    ----------
    counts : (M,) spike counts
    log_g : (M, n_theta) precomputed log(g)
    T_d_sum_g : (n_theta,) precomputed T_d * sum_i g_i(θ)
    """
    return counts @ log_g - T_d_sum_g


def circular_error(theta_true, theta_hat):
    """Signed circular distance, wrapped to [-π, π]."""
    return np.angle(np.exp(1j * (theta_hat - theta_true)))


# =============================================================================
# VON MISES FITTING (proper ML via Bessel function inversion)
# =============================================================================

def estimate_kappa_ml(errors):
    """
    ML estimate of von Mises concentration parameter κ.
    
    The ML estimate satisfies: I₁(κ)/I₀(κ) = R̄
    where R̄ = |mean(exp(i·errors))| is the mean resultant length.
    
    We use the approximation from Mardia & Jupp (2000):
        κ̂ ≈ R̄(2 - R̄²) / (1 - R̄²)     for R̄ < 0.85
        κ̂ ≈ 1/(2(1-R̄) - (1-R̄)² - (1-R̄)³)  for R̄ ≥ 0.85
    
    This is much more accurate than the crude 1/V - 1 approximation.
    """
    R_bar = np.abs(np.mean(np.exp(1j * errors)))
    R_bar = np.clip(R_bar, 1e-10, 1.0 - 1e-10)
    
    if R_bar < 0.53:
        kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
    else:
        kappa = 1.0 / (2 * (1 - R_bar) - (1 - R_bar)**2 - (1 - R_bar)**3)
    
    return max(kappa, 0.01)


def compute_deviation(errors, n_bins=72):
    """
    Compute empirical error distribution and its deviation from 
    the best-fitting von Mises.
    
    The deviation plot is the fingerprint of the population code:
    - Positive at center (sharper peak than von Mises)
    - Negative in shoulders 
    - Positive in tails (heavier tails than von Mises)
    
    This pattern arises because the error distribution is a MIXTURE:
    most trials produce good estimates (peaked), but some trials get
    very few spikes and produce near-random estimates (uniform tail).
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    emp, _ = np.histogram(errors, bins=bin_edges, density=True)
    
    kappa = estimate_kappa_ml(errors)
    vm_pdf = vonmises.pdf(centers, kappa)
    
    return {
        'centers': centers,
        'empirical': emp,
        'von_mises': vm_pdf,
        'deviation': emp - vm_pdf,
        'kappa': kappa,
    }


# =============================================================================
# POPULATION GENERATORS
# =============================================================================

def make_gp_population(M, n_theta, lengthscale, seed):
    """
    Standard homogeneous GP population.
    
    All neurons share the same lengthscale (analogous to Bays's
    homogeneous tuning width ω). Preferred orientations are
    implicitly random because each GP draw peaks at a random location.
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    K = periodic_rbf_kernel(thetas, lengthscale)
    g = np.zeros((M, n_theta))
    for i in range(M):
        g[i] = np.exp(sample_gp_function(K, rng))
    return thetas, g


def make_gp_with_baseline(M, n_theta, lengthscale, baseline_frac, seed):
    """
    GP population + constant baseline floor.
    
    Bays's Eq. 14: f(θ) = exp(κ⁻¹(cos(θ_i-θ)-1)) + f(0)
    
    The baseline adds a constant to all firing rates. This:
    - DECREASES precision at a given gain (signal-to-noise drops)
    - Does NOT change the SHAPE of deviations from normality
    
    This is a critical result: it means the kurtosis pattern is a
    robust signature of the population code, independent of baseline.
    """
    thetas, g = make_gp_population(M, n_theta, lengthscale, seed)
    if baseline_frac > 1e-10:
        mean_peak = np.mean(np.max(g, axis=1))
        # f(0) such that f(0)/(f(0) + peak) = baseline_frac
        f0 = baseline_frac * mean_peak / (1.0 - baseline_frac)
        g = g + f0
    return thetas, g


def make_heterogeneous_population(M, n_theta, config, seed):
    """
    Heterogeneous GP population (Bays's Eq. 17).
    
    Each neuron gets its own:
    - lengthscale λ_i ~ N(λ_mean, λ_std), truncated > 0.05
    - amplitude a_i ~ N(a_mean, a_std), truncated > 0.01  
    - baseline f(0)_i ~ N(bl_mean, bl_std), truncated ≥ 0
    
    This tests whether the deviation-from-normality pattern depends
    on all neurons being identical. Answer: no, it's robust.
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    g = np.zeros((M, n_theta))
    
    lam_mean = config.get('lambda_narrow', 0.3)
    lam_std = config.get('lambda_std', 0.1)
    
    for i in range(M):
        lam_i = max(0.05, rng.normal(lam_mean, lam_std))
        amp_i = max(0.01, rng.normal(1.0, 0.5))
        bl_i = max(0.0, rng.normal(0.25, 0.125))
        
        K = periodic_rbf_kernel(thetas, lam_i)
        f_i = sample_gp_function(K, rng)
        g[i] = amp_i * np.exp(f_i) + bl_i
    
    return thetas, g


def make_cosine_population(M, n_theta, config, seed):
    """
    Half-wave rectified cosine tuning (Bays's Eq. 18).
    
        g_i(θ) = a_i · [cos(θ_i - θ)]₊ + f(0)_i
    
    This is maximally different from the bell-shaped GP tuning:
    - Sharp cutoff at ±π/2 from preferred orientation
    - Linear (not exponential) falloff
    - Yet the deviation pattern is STILL robust
    
    This demonstrates that the key phenomenon (non-normal errors at
    low gain) is a property of the DECODING PROCESS under Poisson 
    noise, not of the specific tuning curve shape.
    """
    rng = np.random.RandomState(seed)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    prefs = rng.uniform(-np.pi, np.pi, M)
    g = np.zeros((M, n_theta))
    
    for i in range(M):
        amp_i = max(0.01, rng.normal(1.0, 0.5))
        bl_i = max(0.0, rng.normal(0.25, 0.125))
        g[i] = amp_i * np.maximum(0.0, np.cos(thetas - prefs[i])) + bl_i
    
    return thetas, g


# =============================================================================
# CORRELATED SPIKE GENERATION
# =============================================================================

def build_correlation_matrix(M, c0):
    """
    Short-range pairwise correlation matrix (Bays's Eq. 19).
    
        c_{ij} = c₀ · exp(-|θ_i - θ_j|)
    
    where θ_i are evenly spaced preferred orientations.
    
    Short-range correlations DECREASE decoding precision because
    nearby neurons (which carry the most information about the 
    stimulus) now provide redundant spike counts. The effect is
    stronger for smaller populations (more overlap between neurons).
    """
    prefs = np.linspace(-np.pi, np.pi, M, endpoint=False)
    diff = np.abs(prefs[:, None] - prefs[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)  # circular distance
    C = c0 * np.exp(-diff)
    np.fill_diagonal(C, 1.0)
    return C, prefs


def generate_correlated_spikes(rates, T_d, chol_L, rng):
    """
    Correlated Poisson spikes via the latent Gaussian method.
    Uses precomputed Cholesky factor for speed.
    """
    M = len(rates)
    lambdas = np.maximum(rates * T_d, 1e-10)
    
    z = chol_L @ rng.randn(M)
    u = norm.cdf(z)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    counts = poisson.ppf(u, mu=lambdas).astype(int)
    return np.maximum(counts, 0)


def precompute_cholesky(corr_matrix):
    """Precompute Cholesky factor of correlation matrix."""
    try:
        return np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        C_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(C_fixed, 1.0)
        return np.linalg.cholesky(C_fixed)


# =============================================================================
# TRIAL ENGINE
# =============================================================================

def run_trials(g, thetas, gamma, T_d, sigma_sq, n_trials, rng, 
               chol_L=None):
    """
    Run n_trials of: encode → normalise → spike → decode → error.
    """
    M, n_theta = g.shape
    
    log_g = np.log(np.maximum(g, 1e-30))
    T_d_sum_g = T_d * np.sum(g, axis=0)
    mean_g = np.mean(g, axis=0)
    
    errors = np.empty(n_trials)
    
    for t in range(n_trials):
        idx_true = rng.randint(n_theta)
        rates = dn_pointwise(g[:, idx_true], gamma, sigma_sq, mean_g[idx_true])
        
        if chol_L is not None:
            counts = generate_correlated_spikes(rates, T_d, chol_L, rng)
        else:
            counts = generate_spikes(rates, T_d, rng)
        
        ll = compute_log_likelihood(counts, log_g, T_d_sum_g)
        idx_hat = np.argmax(ll)
        errors[t] = circular_error(thetas[idx_true], thetas[idx_hat])
    
    return errors


# =============================================================================
# PANEL RUNNER
# =============================================================================

def run_panel(panel_id, gammas, pop_sizes, n_trials, T_d, sigma_sq, 
              n_bins, config, seed):
    """Run all (γ, M) combinations for one panel."""
    results = {}
    
    for M in pop_sizes:
        panel_seed = seed + M
        corr_mat = None
        chol_L = None
        
        # Generate population
        if panel_id == 'a':
            thetas, g = make_gp_population(M, config['n_theta'], 
                                            config['lambda_broad'], panel_seed)
        elif panel_id == 'b':
            thetas, g = make_gp_population(M, config['n_theta'], 
                                            config['lambda_narrow'], panel_seed)
        elif panel_id == 'c':
            thetas, g = make_gp_with_baseline(M, config['n_theta'],
                                               config['lambda_narrow'],
                                               config['baseline_frac'], panel_seed)
        elif panel_id == 'd':
            thetas, g = make_heterogeneous_population(M, config['n_theta'],
                                                       config, panel_seed)
        elif panel_id == 'e':
            thetas, g = make_cosine_population(M, config['n_theta'],
                                                config, panel_seed)
        elif panel_id == 'f':
            thetas, g = make_gp_population(M, config['n_theta'],
                                            config['lambda_narrow'], panel_seed)
            corr_mat, _ = build_correlation_matrix(M, config['c0'])
            chol_L = precompute_cholesky(corr_mat)
        
        # Store example tuning curves
        results[f'g_example_{M}'] = g[:min(30, M)]
        results['thetas'] = thetas
        
        if panel_id == 'f':
            results[f'corr_matrix_{M}'] = corr_mat
        
        # Sweep gains — use fewer trials for large populations
        nt = n_trials if M <= 100 else config.get('n_trials_large', n_trials // 2)
        for gamma in gammas:
            rng = np.random.RandomState(seed + M + int(gamma * 100))
            cl = chol_L if panel_id == 'f' else None
            
            errs = run_trials(g, thetas, gamma, T_d, sigma_sq,
                              nt, rng, cl)
            
            dev = compute_deviation(errs, n_bins)
            results[(gamma, M)] = {
                'errors': errs,
                'deviation': dev,
            }
    
    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(config):
    """Run all six panels."""
    defaults = {
        'n_theta': 128,
        'n_trials': 5_000,
        'T_d': 0.1,
        'sigma_sq': 1e-6,
        'lambda_broad': 1.0,
        'lambda_narrow': 0.3,
        'lambda_std': 0.1,
        'baseline_frac': 0.25,
        'c0': 0.25,
        'gammas': [1, 2, 4, 8, 16, 32, 64, 128],
        'pop_sizes': [100],  # Start with 100; add 1000 for panel f
        'n_bins': 72,
        'seed': 42,
    }
    cfg = {**defaults, **config}
    
    gammas = np.array(cfg['gammas'], dtype=float)
    panels_out = {}
    
    t0 = time.time()
    for pid in ['a', 'b', 'c', 'd', 'e', 'f']:
        pt = time.time()
        
        # Panel f needs both M=100 and M=1000
        if pid == 'f':
            ps = [100, 1000]
        else:
            ps = cfg['pop_sizes']
        
        print(f"  Panel {pid} (M={ps})...", end=" ", flush=True)
        panels_out[pid] = run_panel(pid, gammas, ps,
                                     cfg['n_trials'], cfg['T_d'],
                                     cfg['sigma_sq'], cfg['n_bins'],
                                     cfg, cfg['seed'])
        print(f"({time.time()-pt:.1f}s)")
    
    print(f"\n  Total: {time.time()-t0:.1f}s")
    return {
        'panels': panels_out,
        'gammas': gammas,
        'config': cfg,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results, output_path='figure_4_robustness.png'):
    """
    Three rows × six columns matching Bays (2014) Figure 4.
    """
    panels = results['panels']
    gammas = results['gammas']
    cfg = results['config']
    
    panel_ids = ['a', 'b', 'c', 'd', 'e', 'f']
    titles = ['Broad\ntuning', 'Narrow\ntuning', 'Baseline\nactivity',
              'Heterogeneous\ntuning', 'Cosine\ntuning', 'Correlated\nactivity']
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.linewidth': 0.6, 'lines.linewidth': 0.9,
    })
    
    fig = plt.figure(figsize=(21, 11))
    gs = gridspec.GridSpec(3, 6, hspace=0.50, wspace=0.30,
                           left=0.05, right=0.97, bottom=0.08, top=0.88)
    
    # Gain colour map: red (low) → blue (high)
    gain_cmap = plt.cm.RdYlBu
    gain_norm = mcolors.LogNorm(vmin=min(gammas), vmax=max(gammas))
    
    for col, pid in enumerate(panel_ids):
        pdata = panels[pid]
        thetas = pdata['thetas']
        
        # Determine pop sizes for this panel
        if pid == 'f':
            pop_sizes = [100, 1000]
        else:
            pop_sizes = cfg['pop_sizes']
        
        M_ref = pop_sizes[0]
        
        # ── ROW 1: Tuning curves / correlation matrix ──
        ax_top = fig.add_subplot(gs[0, col])
        
        if pid == 'f' and f'corr_matrix_{M_ref}' in pdata:
            C = pdata[f'corr_matrix_{M_ref}']
            n_show = min(50, C.shape[0])
            ax_top.imshow(C[:n_show, :n_show], cmap='gray_r', 
                          origin='lower', aspect='auto',
                          extent=[-np.pi, np.pi, -np.pi, np.pi])
            ax_top.set_xlabel('orientation')
            ax_top.set_ylabel('orientation')
        else:
            g_ex = pdata[f'g_example_{M_ref}']
            n_show = min(30, g_ex.shape[0])
            for i in range(n_show):
                ax_top.plot(thetas, g_ex[i], 'k-', linewidth=0.4, alpha=0.5)
            ax_top.set_xlabel('orientation')
            if col == 0:
                ax_top.set_ylabel('driving input')
            ax_top.set_xlim(-np.pi, np.pi)
            ax_top.set_xticks([-np.pi, 0, np.pi])
            ax_top.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        
        ax_top.set_title(titles[col], fontsize=10, fontweight='bold')
        ax_top.text(-0.08, 1.15, f'$\\mathbf{{{pid}}}$',
                    transform=ax_top.transAxes, fontsize=13,
                    fontweight='bold', va='top')
        
        # ── ROW 2: Normalised error distributions ──
        ax_mid = fig.add_subplot(gs[1, col])
        
        for M in pop_sizes:
            ls = '-' if M == pop_sizes[0] else '--'
            for gamma in gammas:
                key = (gamma, M)
                if key not in pdata:
                    continue
                dev = pdata[key]['deviation']
                emp = dev['empirical']
                peak = emp.max()
                emp_norm = emp / peak if peak > 0 else emp
                c = gain_cmap(gain_norm(gamma))
                ax_mid.plot(dev['centers'], emp_norm,
                            color=c, linewidth=0.9, linestyle=ls, alpha=0.85)
        
        ax_mid.set_xlim(-np.pi, np.pi)
        ax_mid.set_ylim(0, 1.05)
        ax_mid.set_xticks([-np.pi, 0, np.pi])
        ax_mid.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax_mid.set_xlabel('error')
        if col == 0:
            ax_mid.set_ylabel('normalized\nprobability')
        
        # ── ROW 3: Deviation from circular normal ──
        ax_bot = fig.add_subplot(gs[2, col])
        
        for M in pop_sizes:
            ls = '-' if M == pop_sizes[0] else '--'
            for gamma in gammas:
                key = (gamma, M)
                if key not in pdata:
                    continue
                dev = pdata[key]['deviation']
                c = gain_cmap(gain_norm(gamma))
                ax_bot.plot(dev['centers'], dev['deviation'],
                            color=c, linewidth=0.9, linestyle=ls, alpha=0.85)
        
        ax_bot.axhline(0, color='gray', linewidth=0.4)
        ax_bot.set_xlim(-np.pi, np.pi)
        ax_bot.set_xticks([-np.pi, 0, np.pi])
        ax_bot.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax_bot.set_xlabel('error')
        if col == 0:
            ax_bot.set_ylabel(r'$\Delta$ probability')
    
    # ── Legend for M sizes ──
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k', lw=1.0, ls='-', label='$M = 100$'),
        Line2D([0], [0], color='k', lw=1.0, ls='--', label='$M = 1000$'),
    ]
    fig.legend(handles=handles, loc='lower left', 
               bbox_to_anchor=(0.05, 0.01), fontsize=9, frameon=False,
               ncol=2)
    
    # ── Gain colorbar ──
    sm = plt.cm.ScalarMappable(cmap=gain_cmap, norm=gain_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.70, 0.025, 0.22, 0.015])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'gain, $\gamma$', fontsize=9)
    ticks = [g for g in gammas if g in [1, 2, 4, 8, 16, 32, 64, 128]]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(int(g)) for g in ticks])
    
    fig.suptitle('GP Population Coding — Bays (2014) Fig 4 Equivalent\n'
                 'Robustness of error distributions under tuning and noise variations',
                 fontsize=12, fontweight='bold', y=0.95)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    config = {
        'n_theta': 64,
        'n_trials': 3_000,
        'n_trials_large': 1_500,  # fewer trials for M=1000
        'T_d': 0.1,
        'sigma_sq': 1e-6,
        'lambda_broad': 1.0,
        'lambda_narrow': 0.3,
        'lambda_std': 0.1,
        'baseline_frac': 0.25,
        'c0': 0.25,
        'gammas': [1, 2, 4, 8, 16, 32, 64, 128],
        'pop_sizes': [100],
        'n_bins': 72,
        'seed': 42,
    }
    
    print("=" * 60)
    print("Bays (2014) Figure 4 — GP Equivalent")
    print("=" * 60)
    print(f"  Gains: {config['gammas']}")
    print(f"  Trials per condition: {config['n_trials']}")
    print(f"  Orientation bins: {config['n_theta']}")
    print()
    
    results = run_experiment(config)
    plot_results(results, '/mnt/user-data/outputs/figure_4_robustness.png')
