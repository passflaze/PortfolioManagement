from typing import Any, Dict

import numpy as np
import pandas as pd


def constant_corr_shrinkage(
    returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the constant correlation shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2003) paper "Honey, I Shrunk the Sample Covariance
    Matrix".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    cov_matrix = returns.cov()

    T, N = returns.shape  # T = time periods, N = number of assets

    # Convert to numpy arrays for efficiency
    returns_np = returns.values
    S = cov_matrix.values  # Sample covariance matrix

    # Extract standard deviations
    variances = np.diag(S)
    std_devs = np.sqrt(variances)

    # Compute correlation matrix from covariance matrix
    std_outer = np.outer(std_devs, std_devs)
    corr_matrix = S / std_outer

    # Calculate average correlation
    triu_indices = np.triu_indices_from(corr_matrix, k=1)

    # Calculate the mean
    avg_corr = corr_matrix[triu_indices].mean()

    ## Target
    # Calculate target matrix (constant correlation)
    constant_corr_cov = avg_corr*std_outer  

    # Set diagonal elements to original variances
    np.fill_diagonal(constant_corr_cov, variances)

    target = pd.DataFrame(
        constant_corr_cov, index=cov_matrix.index, columns=cov_matrix.columns
    )

    ## Intensity
    sample_means = np.mean(returns_np, axis=0)

    # Center the returns -> Y has dimensions (T, N)
    Y = returns_np - sample_means

    # Pre-compute matrices for efficiency
    variances = np.diag(S)
    std_devs = np.sqrt(variances)
    sqrt_ratio_matrix = np.outer(std_devs, 1 / std_devs)  # sqrt(S[i,i]/S[j,j])

    # --- START OF VECTORIZATION ---
    
    # Create the 3D tensor of covariance deviations
    # Y[:, :, np.newaxis] has dim (T, N, 1)
    # Y[:, np.newaxis, :] has dim (T, 1, N)
    # Multiplying them yields a tensor (T, N, N) where each "slice" t is the outer product y_t * y_t.T
    outer_prod_tensor = Y[:, :, np.newaxis] * Y[:, np.newaxis, :]
    diff_tensor = outer_prod_tensor - S

    # Pi-hat calculation (Asymptotic variance of the sample covariance matrix)
    # We take the mean over the time axis (axis=0) and sum all elements
    pi_hat = np.sum(np.mean(diff_tensor**2, axis=0))

    # Rho-hat calculation
    # Diagonal terms: (y_it^2 - s_ii)^2
    y_sq_diff = Y**2 - variances # Dimension (T, N)
    rho_hat_diag = np.sum(np.mean(y_sq_diff**2, axis=0))

    # Off-diagonal terms (Theta matrices)
    # We use broadcasting to multiply variance deviations by covariance deviations
    # theta_ii_ij: (T, N, 1) multiplied by (T, N, N) -> mean over T -> (N, N)
    theta_ii_ij = np.mean(y_sq_diff[:, :, np.newaxis] * diff_tensor, axis=0)
    
    # theta_jj_ij: (T, 1, N) multiplied by (T, N, N) -> mean over T -> (N, N)
    theta_jj_ij = np.mean(y_sq_diff[:, np.newaxis, :] * diff_tensor, axis=0)

    # Calculate contribution to rho-hat (off-diagonal elements only)
    off_diag_contrib = (
        0.5 
        * avg_corr 
        * (sqrt_ratio_matrix.T * theta_ii_ij + sqrt_ratio_matrix * theta_jj_ij)
    )

    # Sum only off-diagonal elements by zeroing the diagonal
    np.fill_diagonal(off_diag_contrib, 0)
    rho_hat_off_diag = np.sum(off_diag_contrib)

    # Combine diagonal and off-diagonal parts
    rho_hat = rho_hat_diag + rho_hat_off_diag
    
    # --- END OF VECTORIZATION ---

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target.values - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / (gamma * T)
    numerator = pi_hat - rho_hat
    denominator = gamma_hat * T

    if denominator == 0:
        intensity = 0.0
    else:
        # Ensure shrinkage intensity is clamped between 0 and 1
        intensity = max(0.0, min(1.0, numerator / denominator))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": cov_matrix*(1-intensity) + intensity*target,
    }



def market_factor_shrinkage(
    returns: pd.DataFrame, market_returns: pd.Series
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the market factor shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2002) paper "Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets
        market_returns (pd.Series): The market returns (T x 1)

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    # Align indices to ensure proper calculation
    aligned_data = pd.concat([returns, market_returns], axis=1, join="inner")
    returns_aligned = aligned_data.iloc[:, :-1]
    market_aligned = aligned_data.iloc[:, -1]

    T = returns_aligned.shape[0]

    ## Target
    # Calculate market variance
    market_variance = market_aligned.var()

    # Calculate betas for all assets (vectorized)
    returns_np = returns_aligned.values
    market_np = market_aligned.values

    # Compute covariances between all assets and market at once
    # Stack returns and market, compute covariance matrix, extract asset-market covariances
    combined = np.column_stack([returns_np, market_np])
    cov_matrix_full = np.cov(combined.T)
    cov_with_market = cov_matrix_full[:-1, -1]  # Covariances of each asset with market
    betas = cov_with_market / market_variance  

    # Calculate residual variances: Var(asset) - β² * Var(market) (vectorized)
    asset_variances = np.diag(cov_matrix_full)[:-1]  
    residual_variances = asset_variances - betas**2 * market_variance 

    # Ensure residual variances are positive (vectorized)
    residual_variances = np.maximum(residual_variances, 1e-8)

    # Construct target matrix
    betas_outer = np.outer(betas, betas)
    residual_matrix = np.diag(residual_variances)

    # Final target matrix
    target = pd.DataFrame(
        data=market_variance*betas_outer + residual_matrix, 
        index=returns.columns,
        columns=returns.columns,
    )

    ## Intensity
    cov_matrix = returns_aligned.cov()
    S = cov_matrix.values

    # Center the returns
    sample_means = np.mean(returns_np, axis=0)
    Y = returns_np - sample_means              # (T, N)
    M = market_np - np.mean(market_np)         # (T,)

    # Pre-compute matrices for efficiency
    variances = np.diag(S)
    target_np = target.values

    
    # Prepare base 3D tensors
    # Y_i has dim (T, N, 1), Y_j has dim (T, 1, N)
    Y_i = Y[:, :, np.newaxis]
    Y_j = Y[:, np.newaxis, :]
    
    # outer_prod_tensor is y_it * y_jt for each t
    outer_prod_tensor = Y_i * Y_j              # (T, N, N)
    diff_tensor = outer_prod_tensor - S

    # Pi-hat calculation (Asymptotic variance of S)
    pi_hat = np.sum(np.mean(diff_tensor**2, axis=0))

    # Rho-hat calculation
    # --- Diagonal Part ---
    y_sq_diff = Y**2 - variances               # (T, N)
    rho_hat_diag = np.sum(np.mean(y_sq_diff**2, axis=0))

    # --- Off-Diagonal Part ---
    M_3D = M[:, np.newaxis, np.newaxis]        # (T, 1, 1) - Vectorized market
    
    # betas_y_matrix: y_t[i] * betas[j] -> (T, N, N)
    betas_j = betas[np.newaxis, np.newaxis, :]
    betas_y_matrix_tensor = Y_i * betas_j

    # betas_y_matrix_T: y_t[j] * betas[i] -> (T, N, N)
    betas_i = betas[np.newaxis, :, np.newaxis]
    betas_y_matrix_T_tensor = Y_j * betas_i

    # betas_outer * m_t -> (T, N, N)
    betas_outer_mt = betas_outer[np.newaxis, :, :] * M_3D

    # Calculate the core of the off-diagonal formula for all T and all N, N pairs
    core_term = (
        betas_y_matrix_T_tensor + betas_y_matrix_tensor - betas_outer_mt
    ) * M_3D * outer_prod_tensor

    # Take the mean over T and subtract target_np * S
    off_diag_mat = np.mean(core_term, axis=0) - (target_np * S)

    # Zero out the diagonal and sum
    np.fill_diagonal(off_diag_mat, 0)
    rho_hat_off_diag = np.sum(off_diag_mat)

    # Combine diagonal and off-diagonal
    rho_hat = rho_hat_diag + rho_hat_off_diag
    
    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target_np - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        # Division by T as per your original formula
        intensity = max(0.0, min(1.0, numerator / (T * denominator)))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": (1-intensity) * cov_matrix + intensity * target,
    }