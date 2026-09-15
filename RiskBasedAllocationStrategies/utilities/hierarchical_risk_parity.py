import math
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from typing import Any, Callable, List
from utilities.covariance_utilities import covariance_to_correlation
from utilities.hierarchical_clustering import hierarchical_clustering
from utilities.principal_component_analysis import detone


def correlation_to_hrp_distance(correlation: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to the Lopez de Prado HRP distance matrix
    ``d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))``.

    The result is a proper metric (see lab notes Question 3) and is the input
    expected by the HRP clustering step.

    Parameters:
        correlation (np.ndarray): Correlation matrix.

    Returns:
        np.ndarray: Distance matrix with the same shape as ``correlation``.
    """

    correlation = np.clip(correlation, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - correlation)) # ERROR


def flatten_list(lst: List[Any]) -> List[Any]:
    """
    Recursively flatten a nested list into a single list of elements.

    Parameters:
        lst (List[Any]): The nested list.

    Returns:
        List[Any]: A flattened list of elements.
    """

    if isinstance(lst, list):
        return [item for sub_lst in lst for item in flatten_list(sub_lst)]
    else:
        return [lst]


def is_nested(lst: List[Any]) -> bool:
    """
    Check if a list is nested (contains other lists as elements).

    Parameters:
        lst (List[Any]): The list to check.

    Returns:
        bool: True if the list is nested, False otherwise.
    """

    return any(isinstance(element, list) for element in lst)


def list_recursive_bisection(
    lst: List[Any],
    labels: List[Any] | None = None,
    cur_iter: int | None = None,
    max_iter: int | None = None,
) -> List[Any]:
    """
    Perform a recursive bisection of a list.

    Parameters:
        lst (List[Any]): The original list to be bisected.
        labels (List[Any] | None): Labels, default to None.
        cur_iter (int | None): Current iteration, default to None.
        max_iter (int | None): Maximum number of iterations, default to None.

    Returns:
        list: A nested list representing the recursive bisection.
    """
    # Stop conditions: list of length <= 1, or iteration limit reached
    if len(lst) <= 1:
        return lst
    if max_iter is not None and cur_iter is not None and cur_iter >= max_iter:
        return lst

    # Halve the list at the midpoint
    mid = len(lst) // 2
    left = lst[:mid]
    right = lst[mid:]

    # Track recursion depth (only when a max_iter ceiling was specified)
    next_iter = None if cur_iter is None else cur_iter + 1

    return [
        list_recursive_bisection(left,  labels, next_iter, max_iter),
        list_recursive_bisection(right, labels, next_iter, max_iter),
    ]

def recursive_bisection(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Return the nested cluster structure from a linkage matrix performing a recursive bisection.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """

    iter_num = None if clusters_num is None else math.log(clusters_num, 2)
    if iter_num is not None:
        if not iter_num.is_integer():
            raise ValueError(
                "The number of clusters must be a power of 2 for recursive bisection."
            )
        else:
            iter_num = int(iter_num)

    leaves_lst = sch.leaves_list(linkage_matrix.values)
    leaves_labels = (
        list(leaves_lst) if labels is None else [labels[leaf] for leaf in leaves_lst]
    )

    return list_recursive_bisection(
        lst=leaves_labels,
        labels=labels,
        cur_iter=0 if iter_num is not None else None,
        max_iter=iter_num,
    )


def dendrogram_iteration(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Convert a linkage matrix to nested clusters according to the dendrogram structure.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """

    n_leaves = linkage_matrix.shape[0] + 1
    n_merges = linkage_matrix.shape[0]

    if clusters_num is not None:
        if clusters_num < 1 or clusters_num > n_leaves:
            raise ValueError(
                f"clusters_num must be in [1, {n_leaves}], got {clusters_num}"
            )

    # Resolve labels: if None, use 0..n_leaves-1
    leaf_labels = (
        list(range(n_leaves)) if labels is None else list(labels)
    )

    # Build the cluster contents bottom-up.
    # cluster_contents[k] is the nested-list representation of cluster k.
    # Clusters 0..n_leaves-1 are the original assets (singletons).
    # Cluster n_leaves+i is formed at row i of the linkage matrix.
    cluster_contents: dict[int, Any] = {
        i: [leaf_labels[i]] for i in range(n_leaves)
    }
    cluster_distance: dict[int, float] = {i: 0.0 for i in range(n_leaves)}

    Z = linkage_matrix.values
    for i in range(n_merges):
        c1 = int(Z[i, 0])
        c2 = int(Z[i, 1])
        d = float(Z[i, 2])
        new_id = n_leaves + i
        cluster_contents[new_id] = [cluster_contents[c1], cluster_contents[c2]]
        cluster_distance[new_id] = d

    # The root cluster is the last one merged
    root_id = n_leaves + n_merges - 1

    # Top-down split: keep splitting the cluster with the largest merge distance
    # until we have `clusters_num` clusters (or all leaves are singletons).
    if clusters_num is None:
        target_clusters = n_leaves    # split everything down to leaves
    else:
        target_clusters = clusters_num

    # Active clusters: list of (cluster_id, contents) tuples.
    # We'll repeatedly find the one with largest merge distance and split it.
    active = [(root_id, cluster_contents[root_id])]

    while len(active) < target_clusters:
        # Find the active cluster with the largest merge distance that is splittable
        splittable = [
            (idx, cid) for idx, (cid, _) in enumerate(active)
            if cid >= n_leaves   # cluster_id < n_leaves means singleton, can't split
        ]
        if not splittable:
            break  # everything is a singleton

        # Pick the one with largest merge distance
        idx_to_split, cid_to_split = max(
            splittable, key=lambda x: cluster_distance[x[1]]
        )

        # Find its children from the linkage matrix
        merge_row = cid_to_split - n_leaves
        c1 = int(Z[merge_row, 0])
        c2 = int(Z[merge_row, 1])

        # Replace the active cluster with its two children
        active.pop(idx_to_split)
        active.insert(idx_to_split,     (c2, cluster_contents[c2]))
        active.insert(idx_to_split,     (c1, cluster_contents[c1]))

    # If clusters_num was None, we kept splitting until all singletons —
    # but that loses the binary tree shape. Instead, if clusters_num is None
    # we want the full nested binary structure: just return the root contents.
    if clusters_num is None:
        return cluster_contents[root_id]

    # Otherwise return the flat list of cluster contents at the requested granularity
    return [c for _, c in active]

def top_down_allocation(
    nested_clusters: List[Any], covariance: pd.DataFrame, get_cluster_var: Callable
) -> pd.Series:
    """
    Top-down allocation of weights to the clusters following Lopez de Prado's
    recursive split: at each bisection ``alpha = sigma2_R / (sigma2_L + sigma2_R)``
    is allocated to the left cluster, ``1 - alpha`` to the right one (more capital
    to the lower-variance branch). The variance of each cluster is computed via
    ``get_cluster_var``.

    Parameters:
        nested_clusters (List[Any]): The nested cluster structure produced by
            ``recursive_bisection`` or ``dendrogram_iteration``.
        covariance (pd.DataFrame): Covariance matrix.
        get_cluster_var (Callable): Function returning the variance of a cluster
            given the covariance matrix and the list of asset labels in the
            cluster.

    Returns:
        pd.Series: Weights summing to 1, indexed on the asset labels.
    """

    weights = pd.Series(1.0, index=flatten_list(nested_clusters))
    if not is_nested(nested_clusters):
        return weights
    else:
        cluster1 = flatten_list(nested_clusters[0])
        cluster2 = flatten_list(nested_clusters[1])

        cluster1_var = get_cluster_var(covariance=covariance, cluster=cluster1)
        cluster2_var = get_cluster_var(covariance=covariance, cluster=cluster2)

        # Inverse-variance split: more capital to the lower-variance branch
        alpha1 = cluster2_var / (cluster1_var + cluster2_var)
        alpha2 = 1.0 - alpha1

        weights[cluster1] *= alpha1 * top_down_allocation(
            nested_clusters[0], covariance.loc[cluster1, cluster1], get_cluster_var
        )
        weights[cluster2] *= alpha2 * top_down_allocation(
            nested_clusters[1], covariance.loc[cluster2, cluster2], get_cluster_var
        )

        return weights


def get_cluster_var_via_iv(covariance: pd.DataFrame, cluster: List[Any]):

    covariance = covariance.loc[cluster, cluster]
    iv_weights = 1.0 / np.diag(covariance)
    iv_weights /= iv_weights.sum()
    iv_weights = iv_weights.reshape(-1, 1)

    return (iv_weights.T @ covariance.values @ iv_weights)[0, 0]


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    cluster_traverser: Callable = dendrogram_iteration,
    get_cluster_var: Callable = get_cluster_var_via_iv,
    perform_detoning: bool = False,
    plot_dendrogram: bool = False,
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity weights for a single rebalance.

    Pipeline:
        1. Convert ``covariance`` to a correlation matrix.
        2. Optionally detone the correlation matrix by removing its first
           principal component.
        3. Build a linkage matrix on the HRP distance ``sqrt(0.5(1 - rho))``
           using the chosen linkage method and (second-stage) distance metric.
        4. Convert the linkage to a nested cluster structure via
           ``cluster_traverser`` (e.g. ``recursive_bisection`` or
           ``dendrogram_iteration``).
        5. Allocate weights top-down with inverse-variance splits.

    Parameters:
        covariance (pd.DataFrame): Covariance matrix indexed by asset labels.
        linkage_method (str): Linkage method passed to ``scipy.linkage``
            (``single``, ``ward``, ``complete``, ``average``, ...).
        distance_metric (str): Metric used by ``scipy.linkage`` to compute
            distances between rows of the input matrix (defaults to
            ``euclidean``, following Lopez de Prado).
        cluster_traverser (Callable): Function mapping a linkage matrix to a
            nested cluster structure. Defaults to ``dendrogram_iteration``.
        get_cluster_var (Callable): Function returning the variance of a
            cluster. Defaults to inverse-variance parity on the diagonal.
        perform_detoning (bool): Remove the first principal component from the
            correlation matrix before clustering. Defaults to False.
        plot_dendrogram (bool): Whether to plot the dendrogram while building it.

    Returns:
        pd.Series: HRP weights summing to 1, indexed on the asset labels.
    """
    correlation = covariance_to_correlation(covariance=covariance.values)
    if perform_detoning:
        correlation = detone(corr_matrix=correlation, components_num=1)
    distance = correlation_to_hrp_distance(correlation)
    linkage_matrix = hierarchical_clustering(
        matrix=pd.DataFrame(
            distance, index=covariance.index, columns=covariance.columns
        ),
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        plot_dendrogram=plot_dendrogram,
    )

    nested_clusters = cluster_traverser(
        linkage_matrix, labels=covariance.index.tolist()
    )

    return top_down_allocation(
        nested_clusters=nested_clusters,
        covariance=covariance,
        get_cluster_var=get_cluster_var,
    )
