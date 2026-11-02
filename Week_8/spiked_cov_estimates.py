import numpy as np
import pandas as pd

def simulate_correlated_prices(Sigma=None, T=1000):
    """
    Simulate correlated price series using Cholesky decomposition
    """
    # Step 1: Define theoretical covariance matrix

    n = Sigma.shape[0]

    print("Theoretical Covariance Matrix:")
    print(Sigma)
    print()

    # Step 2: Cholesky decomposition
    L = np.linalg.cholesky(Sigma)
    # print("Cholesky Factor L:")
    # print(L)
    # print()

    # Step 3: Generate correlated returns
    np.random.seed(43)  # For reproducibility

    # Generate independent standard normal random variables
    Z = np.random.standard_normal((T, n))

    # Transform to correlated returns

    returns = Z @ L.T  # Each row is a day, each column is an asset

    # Step 4: Convert to price series
    P0 = 100  # Initial price
    log_prices = np.cumsum(returns / 100, axis=0)  # Cumulative log returns (returns are in percent)
    prices = P0 * np.exp(log_prices)

    # Add initial price
    prices = np.vstack([np.full(n, P0), prices])

    return prices, returns, Sigma

def calculate_empirical_covariance(returns):
    """
    Calculate empirical covariance matrix from returns
    """
    return np.cov(returns, rowvar=False)


def compare_covariances(theoretical, empirical, is_pca):
    """
    Compare theoretical and empirical covariance matrices
    """

    if is_pca:
        prefix = "PCA"
    else:
        prefix = "Empirical"
    print(f"{prefix} Covariance Matrix:")
    print(empirical)
    print()

    # Calculate differences
    diff = theoretical - empirical
    if is_pca:
        print("Difference (Theoretical - PCA):")
    else:
        print("Difference (Theoretical - Empirical):")
    print(diff)
    print()

    # Frobenius norm
    frobenius_norm = np.linalg.norm(diff, 'fro')
    theoretical_norm = np.linalg.norm(theoretical, 'fro')
    relative_error = frobenius_norm / theoretical_norm
    print(f"Frobenius Norm of Difference: {frobenius_norm:.6f}")
    print(f"Frobenius Norm of Theoretical: {theoretical_norm:.6f}")
    print(f"Relative Frobenius Error:      {relative_error:.4%}")
    print()

    return diff, frobenius_norm, relative_error

def calculate_pca_covariance(returns, n_factors=None, two_stage=False, show_variance_explained=False):
    """
    Simulate returns from theoretical covariance matrix and calculate PCA-based covariance estimate

    Parameters:
    -----------
    returns : np.ndarray
        T x N matrix of simulated returns        
    n_factors : int, optional
        Number of principal components to use. If None, uses all components.
    two_stage : bool, default=False
        If True, uses two-stage estimation (factor model + residual covariance)
        If False, uses one-stage estimation (reconstruct from top k factors only)
        

    Returns:
    --------
    Sigma_pca : np.ndarray
        N x N covariance matrix estimated using PCA
    """

    # Calculate empirical covariance
    Sigma_emp = np.cov(returns, rowvar=False)

    n = Sigma_emp.shape[0]    

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_emp)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Determine number of factors
    if n_factors is None:
        n_factors = n
    else:
        n_factors = min(n_factors, n)

    if show_variance_explained:
        total_var = eigenvalues.sum()
        individual = eigenvalues[:n_factors] / total_var
        cumulative = np.cumsum(individual)
        print(f"Variance Explained by first {n_factors} PCs:")
        print(f"{'k':>5} {'lambda_k':>12} {'indiv %':>10} {'cum %':>10}")
        for k in range(n_factors):
            print(f"{k+1:>5} {eigenvalues[k]:>12.6f} {individual[k]:>9.4%} {cumulative[k]:>9.4%}")
        print()

    if two_stage:
        # Two-stage estimation: Factor model + diagonal residual
        # Sigma = B * F * B' + D
        # where B = loadings (eigenvectors), F = factor covariance, D = residual variance

        # Factor loadings (top k eigenvectors)
        B = eigenvectors[:, :n_factors]

        # Factor variances (top k eigenvalues)
        F = np.diag(eigenvalues[:n_factors])

        # Factor contribution to covariance
        Sigma_factors = B @ F @ B.T

        # Residual covariance (diagonal only)
        residuals = returns - returns @ B @ B.T
        D = np.diag(np.var(residuals, axis=0))

        # Total covariance
        Sigma_pca = Sigma_factors + D

    else:
        # One-stage estimation: Reconstruct using top k factors only
        # Sigma ≈ sum_{i=1}^k lambda_i * v_i * v_i'

        Sigma_pca = np.zeros((n, n))
        for i in range(n_factors):
            Sigma_pca += eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i])

    return Sigma_pca


def generate_random_covariance(n=4):

    A = .1 * np.random.randn(n, n)
    Sigma = A.T @ A + 1e-6 * np.eye(n)
    return Sigma


def generate_factor_covariance(n=1500, k=5, factor_vars=None, idio_var=0.5):
    """
    Covariance with genuine factor structure: Sigma = B F B' + D.
    B: N x k loadings (iid N(0,1)), F: diagonal factor variances,
    D: diagonal idiosyncratic variance. PCA is designed for this regime.

    Default factor variances mimic equity spectra: daily returns in percent (~1% vol), one dominant market
    factor (0.25), then k-1 weaker factors decaying geometrically from 0.03.
    """
    if factor_vars is None:
        decay = 0.9
        factor_vars = np.concatenate([[0.25], 0.03 * decay ** np.arange(k - 1)])
    factor_vars = np.asarray(factor_vars)
    if len(factor_vars) != k:
        raise ValueError(f"factor_vars has {len(factor_vars)} entries, expected k={k}")
    B = np.random.randn(n, k)
    F = np.diag(factor_vars)
    Sigma = B @ F @ B.T + idio_var * np.eye(n)
    return Sigma
    
    
def main():
    """
    Main function to run the covariance simulation
    """
    print("Covariance Matrix Simulation")
    print("=" * 40)

    np.set_printoptions(precision=4, linewidth=240, suppress=True)

    N = 5000
    T = 252 * 10
    N_FACTORS = 25
    K_TRUE = 100
    two_stage = True

    # Simulate data
    # prices, returns, Sigma_theoretical = simulate_correlated_prices(T=T)
    Sigma = generate_factor_covariance(N, k=K_TRUE)
    prices, returns, Sigma_theoretical = simulate_correlated_prices(Sigma=Sigma, T=T)

    # Calculate empirical covariance
    Sigma_empirical = calculate_empirical_covariance(returns)
    Sigma_pca = calculate_pca_covariance(returns, n_factors=N_FACTORS, two_stage=two_stage, show_variance_explained=True)

    # Compare matrices
    diff, norm, rel_err = compare_covariances(Sigma_theoretical, Sigma_pca, is_pca=True)
    diff, norm, rel_err = compare_covariances(Sigma_theoretical, Sigma_empirical, is_pca=False)
    
if __name__ == "__main__":
    main()
