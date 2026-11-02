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
    log_prices = np.cumsum(returns, axis=0)  # Cumulative log returns
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
    print(f"Frobenius Norm of Difference: {frobenius_norm:.6f}")
    print()

    return diff, frobenius_norm

def calculate_pca_covariance(returns, n_factors=None, two_stage=False):
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

    A = .1 * np.random.randn(2 * n, n)
    Sigma = A.T @ A
    return np.round(Sigma, 4)
    
    
def main():
    """
    Main function to run the covariance simulation
    """
    print("Covariance Matrix Simulation")
    print("=" * 40)

    np.set_printoptions(precision=4, linewidth=240, suppress=True)

    N = 5000
    T = 252 * 10
    N_FACTORS = 20

    # Simulate data
    # prices, returns, Sigma_theoretical = simulate_correlated_prices(T=T)
    Sigma = generate_random_covariance(N)
    prices, returns, Sigma_theoretical = simulate_correlated_prices(Sigma=Sigma, T=T)

    # Calculate empirical covariance
    Sigma_empirical = calculate_empirical_covariance(returns)
    Sigma_pca = calculate_pca_covariance(returns, n_factors=N_FACTORS, two_stage=False)

    # Compare matrices
    diff, norm = compare_covariances(Sigma_theoretical, Sigma_pca, is_pca=True)
    diff, norm = compare_covariances(Sigma_theoretical, Sigma_empirical, is_pca=False)
    
if __name__ == "__main__":
    main()
