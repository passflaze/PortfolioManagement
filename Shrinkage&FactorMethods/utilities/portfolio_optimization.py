"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)

References:
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

import numpy as np
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1), i.e. the solution of the optimization problem: ## QUI C'ERA UN ERRORE
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    inv_cov_matrix = np.linalg.inv(cov_matrix)

    min_var_ptf_numerator = inv_cov_matrix @ ones_vec
    min_var_ptf_denominator = ones_vec.T @ inv_cov_matrix @ ones_vec

    min_var_ptf_weights = min_var_ptf_numerator / min_var_ptf_denominator

    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w , subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )
    

    # Ensure expected_returns is a column matrix 
    mu = expected_returns.reshape(-1, 1)
    
    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    inv_cov = np.linalg.inv(cov_matrix)
    inv_cov_mu = inv_cov @ mu
    inv_cov_ones = inv_cov @ ones_vec

    # Calculate A and C as scalars using .item()
    
    A = (ones_vec.T @ inv_cov_mu).item()
    C = (ones_vec.T @ inv_cov_ones).item()

    if A == 0:
        raise ValueError("A cannot be zero")

    term1 = (1 / risk_aversion) * inv_cov_mu
    term2 = ((1 - (A / risk_aversion)) / C) * inv_cov_ones
    
    mean_var_ptf_weights = term1 + term2

    return mean_var_ptf_weights.flatten()
