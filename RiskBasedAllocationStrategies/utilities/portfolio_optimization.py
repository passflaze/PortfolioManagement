"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)
- Equal risk contribution portfolio (optimization-based)

References:
- Maillard, S., Roncalli, T., & Teïletche, J. (2010).
  "The properties of equally weighted risk contribution portfolios."
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

from typing import Any
import numpy as np
from functools import partial
from scipy.optimize import minimize
from scipy import optimize as opt
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
    risk_contribution,
)



def inverse_volatility_portfolio(covariance: np.ndarray) -> np.ndarray:
    """
    Compute the inverse-volatility (naive risk-parity) portfolio.

    Each weight is proportional to the inverse of the asset's volatility,
    ``w_i ∝ 1 / sigma_i``, with ``sum_i w_i = 1``. Lab notes Question 2 shows
    that this coincides with the ERC solution when the assets are uncorrelated.

    Parameters:
        covariance (np.ndarray): Covariance matrix of asset returns. Only the
            diagonal is used.

    Returns:
        np.ndarray: Inverse-volatility weights summing to 1.
    """

    covariance = _validate_covariance_matrix(
        covariance,
        name="covariance"
    )

    variances = np.diag(covariance)
    if np.any(variances <= 0):
        raise ValueError(
            "covariance matrix has non-positive variances on the diagonal; "
            "check the covariance matrix"
        )
    inverse_vol = 1.0 / np.sqrt(variances)
    return inverse_vol / np.sum(inverse_vol)

    

def erc_objective_function(weights: np.ndarray, covariance: np.ndarray) -> float:
    """
    Equal risk contribution objective function implemented as the variance of the risk
    contributions. Minimizing this function leads to equal risk contributions across assets.

    Parameters:
        weights (np.ndarray): Portfolio weights.
        covariance (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Objective function value.
    """

    non_normalized_risk_contributions = (
        np.multiply(weights.dot(covariance), weights)
    ).reshape(-1, 1)

    return len(non_normalized_risk_contributions) * np.sum(
        np.square(non_normalized_risk_contributions)
    ) - np.sum(non_normalized_risk_contributions @ non_normalized_risk_contributions.T)


def equal_risk_contribution_portfolio(
    covariance: np.ndarray,
    initial_solution: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
    pcr_tolerance: float = 0.001,
    ignore_objective: bool = False,
) -> np.ndarray:
    """
    Calculate the equal risk contribution portfolio.

    Parameters:
        covariance (np.ndarray): Covariance matrix of assets, must be positive definite.
        initial_solution (np.ndarray | None): Initial solution guess, default to
            None, i.e. to the inverse volatility portfolio.
        options (Dict[str, Any] | None): A dictionary of solver options, see
            scipy.optimize.minimize.
        pcr_tolerance (float): The max allowable tolerance for differences in the percentage
            contribution to risk (pcr) coming from different assets, default to 10bps.

    Returns:
        np.ndarray: Equal risk contribution portfolio.
    """

    covariance = _validate_covariance_matrix(
        covariance,
        name="covariance",
        require_positive_definite=True,
        positive_definite_message=(
            "covariance must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n_assets = covariance.shape[0]

    if initial_solution is None:
        initial_solution = inverse_volatility_portfolio(covariance)
    else:
        initial_solution = np.asarray(initial_solution, dtype=float).flatten()
        if initial_solution.shape[0] != n_assets:
            raise ValueError(
                f"initial_solution has {initial_solution.shape[0]} assets "
                f"but covariance has {n_assets}"
            )

    # Long-only, fully invested
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Sensible solver defaults; user can override
    default_options = {"ftol": 1e-12, "maxiter": 500, "disp": False}
    if options is not None:
        default_options.update(options)

    result = opt.minimize(
        fun=erc_objective_function,
        x0=initial_solution,
        args=(covariance,),    
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=default_options,
    )

    weights = result.x

    # Renormalize defensively — SLSQP usually respects the equality
    # constraint to ~1e-10, but tiny drift can hurt downstream checks.
    weights = weights / weights.sum()

    
    if not ignore_objective:
        rc = risk_contribution(weights.reshape(1, -1), covariance).flatten()
        portfolio_vol = rc.sum()                      # = sqrt(w' Σ w) by Euler
        pcr = rc / portfolio_vol                      # shares, sum to 1

        if not np.all(np.abs(pcr - 1.0 / n_assets) < pcr_tolerance):
            print(
                f"Warning: ERC optimizer did not converge to within tolerance. "
                f"Solver message: {result.message}"
            )
    

    return weights


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

