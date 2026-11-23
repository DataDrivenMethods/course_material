"""
Mean-variance portfolio optimizer (pure Python, OSQP).

Solves the standard mean-variance problem
    max_w  alpha' w  -  (risk_aversion / 2) w' Sigma w  -  l1_penalty * sum(w_i^2)
i.e.
    min_w  0.5 w' (risk_aversion * Sigma + 2 * l1_penalty * I) w  -  alpha' w

risk_aversion is the usual gamma: larger means more risk-averse. Unconstrained
(up to the budget constraint) the solution is w ~ Sigma^{-1} alpha / gamma.

Long-only:   sum(w) = 1,  0 <= w_i <= max_position
Long-short:  sum(w) = 0,  -max_position <= w_i <= max_position,
             sum(|w_i|) <= gross_leverage  (if gross_leverage > 0)

Note: despite its name, l1_penalty is a quadratic (L2) concentration
penalty, l1_penalty * sum(w_i^2).

Originally a port of a C++/SWIG OSQP module, which multiplied alpha by
risk_aversion instead. Same solver (OSQP) and settings. Requires: numpy,
pandas, scipy, osqp.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import osqp


def _solve_qp(alpha, cov, risk_aversion, long_only, max_position,
              gross_leverage, l1_penalty):
    n = len(alpha)

    if not long_only and gross_leverage < 0:
        gross_leverage = 2.0
    use_gross = (not long_only) and gross_leverage > 0

    # Objective: 0.5 w' (gamma Sigma + 2 l I) w - alpha' w
    P_w = sp.csc_matrix(risk_aversion * cov + 2.0 * l1_penalty * np.eye(n))
    q_w = -alpha
    I = sp.identity(n, format="csc")
    ones = sp.csc_matrix(np.ones((1, n)))

    if use_gross:
        # Variables [w, z] with z_i >= |w_i|
        Z = sp.csc_matrix((n, n))
        P = sp.block_diag([P_w, Z], format="csc")
        q = np.concatenate([q_w, np.zeros(n)])
        A = sp.vstack([
            sp.hstack([ones, sp.csc_matrix((1, n))]),   # sum(w) = 0
            sp.hstack([I, Z]),                          # |w_i| <= max_position
            sp.hstack([-I, I]),                         # z - w >= 0
            sp.hstack([I, I]),                          # z + w >= 0
            sp.hstack([sp.csc_matrix((1, n)), ones]),   # sum(z) <= gross
        ], format="csc")
        l = np.concatenate([[0.0], -max_position * np.ones(n),
                            np.zeros(n), np.zeros(n), [-np.inf]])
        u = np.concatenate([[0.0], max_position * np.ones(n),
                            np.inf * np.ones(n), np.inf * np.ones(n),
                            [gross_leverage]])
    else:
        P, q = P_w, q_w
        A = sp.vstack([ones, I], format="csc")
        if long_only:
            l = np.concatenate([[1.0], np.zeros(n)])
            u = np.concatenate([[1.0], max_position * np.ones(n)])
        else:
            l = np.concatenate([[0.0], -max_position * np.ones(n)])
            u = np.concatenate([[0.0], max_position * np.ones(n)])

    solver = osqp.OSQP()
    solver.setup(sp.triu(P, format="csc"), q, A, l, u,
                 verbose=False, alpha=1.6, eps_abs=1e-5, eps_rel=1e-5,
                 max_iter=10000)
    res = solver.solve(raise_error=False)

    if res.info.status in ("solved", "solved inaccurate") and res.x is not None:
        return np.asarray(res.x[:n], dtype=float), 0

    fallback = np.full(n, 1.0 / n) if long_only else np.zeros(n)
    return fallback, -1


def optimize_portfolio(alphas, cov_matrix, risk_aversion=1.0,
                       long_only=True, max_position=0.05,
                       gross_leverage=None, l1_penalty=0.0, min_positions=None):
    """
    Mean-variance portfolio optimizer.

    Args:
        alphas: Expected returns (array, Series or one-column DataFrame)
        cov_matrix: Covariance matrix (n x n)
        risk_aversion: Risk aversion gamma (larger = more risk-averse)
        long_only: If True, w >= 0 and sum(w) = 1. Else sum(w) = 0
        max_position: Maximum absolute position size
        gross_leverage: Max sum of |w| (long-short only, default 2.0)
        l1_penalty: Quadratic concentration penalty l1_penalty * sum(w^2)
        min_positions: Not implemented

    Returns:
        weights: Portfolio weights (array)
        opt_value: Objective value (minimised form)
        status: "optimal" or "failed"
    """
    if isinstance(alphas, pd.DataFrame):
        alpha_array = pd.to_numeric(alphas.iloc[:, 0], errors='coerce').values
    elif isinstance(alphas, pd.Series):
        alpha_array = pd.to_numeric(alphas, errors='coerce').values
    else:
        alpha_array = np.array(alphas, dtype=float).flatten()
    alpha_array = np.where(np.isnan(alpha_array), 0.0, alpha_array).astype(float)

    cov_array = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) \
        else np.array(cov_matrix, dtype=float)

    if gross_leverage is None:
        gross_leverage = 2.0 if not long_only else -1.0

    weights, result = _solve_qp(alpha_array, cov_array, risk_aversion,
                                long_only, max_position, gross_leverage,
                                l1_penalty)

    risk = 0.5 * risk_aversion * weights @ cov_array @ weights
    expected_return = alpha_array @ weights
    opt_value = risk - expected_return
    if l1_penalty > 0:
        opt_value += l1_penalty * np.sum(weights ** 2)

    status = "optimal" if result == 0 else "failed"
    return weights, opt_value, status


if __name__ == "__main__":
    import time
    np.random.seed(42)
    n = 100
    alphas = np.random.randn(n) * 0.01
    A = np.random.randn(n, n) * 0.1
    cov_matrix = A @ A.T + np.eye(n) * 0.01

    start = time.time()
    weights, val, status = optimize_portfolio(
        alphas, cov_matrix, risk_aversion=1.0, long_only=True, max_position=0.05)
    elapsed = (time.time() - start) * 1000

    print(f"Status: {status}")
    print(f"Time: {elapsed:.2f} ms")
    print(f"Sum: {weights.sum():.6f}")
    print(f"Max: {weights.max():.6f}")
