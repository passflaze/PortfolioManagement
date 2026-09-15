import numpy as np
import pandas as pd
from utilities.covariance_utilities import covariance_to_correlation


def principal_component_analysis(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a matrix, returns the eigenvalues vector and the eigenvectors matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sorting from greatest to lowest the eigenvalues and the eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def detone(corr_matrix: np.ndarray, components_num: int = 1) -> np.ndarray:
    """
    Remove components_num principal components from an input matrix.

    Parameters:
        corr_matrix (np.ndarray): Correlation matrix to be detoned.
        components_num (int): Components number to remove, default to one.

    Returns:
        np.ndarray: Detoned correlation matrix.
    """

    if components_num < 1:
        raise ValueError(
            f"components_num must be >= 1, got {components_num}"
        )
    if components_num >= corr_matrix.shape[0]:
        raise ValueError(
            f"components_num ({components_num}) must be smaller than the "
            f"matrix dimension ({corr_matrix.shape[0]})"
        )

    eig_vals, eig_vecs = principal_component_analysis(corr_matrix)

    # Take the top components_num eigenvectors and eigenvalues
    top_eig_vecs = eig_vecs[:, :components_num]
    top_eig_vals = eig_vals[:components_num]

    # Reconstruct the contribution of the top components
    tone = ( top_eig_vecs * top_eig_vals ) @ top_eig_vecs.T

    # Subtract the dominant component(s) from the original correlation matrix
    detoned_corr = corr_matrix - tone

    # Rescale so the diagonal is 1 again — required for it to remain a
    # valid correlation matrix (in the unit-diagonal sense).
    diag = np.sqrt(np.diag(detoned_corr))
    if np.any(diag <= 0):
        raise ValueError(
            "Detoned matrix has non-positive diagonal entries; "
            "cannot rescale to a correlation matrix"
        )
    detoned_corr = covariance_to_correlation(detoned_corr)

    return detoned_corr


def align_eigenvectors_to_previous(
    current_eig_vecs: pd.DataFrame,
    previous_eig_vecs: pd.DataFrame | None,
) -> pd.DataFrame:
    aligned = current_eig_vecs.copy()

    if previous_eig_vecs is None:
        if aligned.iloc[:, 0].sum() < 0:
            aligned.iloc[:, 0] *= -1
        return aligned

    common_assets = aligned.index.intersection(previous_eig_vecs.index)
    components_num = min(
        len(common_assets),
        aligned.shape[1],
        previous_eig_vecs.shape[1],
    )
    for pc in range(components_num):
        similarity = float(
            aligned.loc[common_assets, pc].dot(previous_eig_vecs.loc[common_assets, pc])
        )
        if similarity < 0:
            aligned.iloc[:, pc] *= -1

    return aligned


def pca_denoise_covariance(
    cov_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise a covariance matrix by keeping k signal eigenvalues and averaging the rest.

    Args:
        cov_matrix: Sample covariance matrix (N x N).
        k: Number of principal components to retain as signal.

    Returns:
        Denoised covariance matrix and its eigenvalues (sorted descending).
    """
    eig_vals, eig_vecs = principal_component_analysis(cov_matrix)
    noise_avg = np.mean(eig_vals[k:])
    denoised_eig_vals = np.concatenate(
        [eig_vals[:k], np.full(len(eig_vals) - k, noise_avg)]
    )
    denoised_cov = (eig_vecs * denoised_eig_vals) @ eig_vecs.T
    return denoised_cov, denoised_eig_vals
