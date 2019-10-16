'''Kernelized General Fuzzy c-Means Clustering

Paper Source: Gupta A., Das S., 'On the Unification of k-Harmonic Means and
Fuzzy c-Means Clustering Problems under Kernelization', in the 2017 Ninth
International Conference on Advances in Pattern Recognition (ICAPR 2017),
pp. 386-391, 2017.
'''

# Author: Avisek Gupta


import numpy as np
from scipy.spatial.distance import cdist


def kernel_gfcm(
    X, n_clusters, sigma=1, m=2, p=2, max_iter=300,
    n_init=30, tol=1e-16
):
    '''Kernelized General Fuzzy c-Means Clustering

    Notes
    -----
    Set p=2 for Kernel Fuzzy c-Means.
    Set m=2 for Kernel k-Harmonic Means.

    Parameters
    ----------

    X : array, shape (n_data_points, n_features)
        The data array.

    n_clusters : int
        The number of clusters.

    sigma : float
        Parameter for the Gaussian Kernel.
        K(a,b) = np.exp(-(a-b)**2 / (2 * (sigma ** 2)))

    m : float, default: 2
        Level of fuzziness. Set m=2 for Kernel k-Harmonic Means.

    p: float, default: 2
        Power of Euclidean distance. Set p=2 for Kernel Fuzzy c-Means.

    max_iter: int, default: 300
        The maximum number of iterations of the KGFCM algorithm.

    n_init: int, default 30
        The number of runs of the KGFCM algorithm. The results corresponding
        to the minimum cost will be returned.

    tol: float, default 1e-16
        Tolerance for convergence.

    Returns
    -------

    mincost_centers: array, shape (n_clusters, n_features)
        The resulting cluster centers.

    mincost_mem: array, shape (n_data_points)
        The resulting cluster memberships.

    min_cost: float
        The lowest cost achieved in n_init runs.

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([
    ...    [10, 11], [12, 13], [14, 15], [-10, -11], [-12, -13], [-14, -15]
    ... ])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> k = 2
    >>> centers, mem, cost = kernel_gfcm(X, n_clusters=k)
    >>> print(centers)
    [[ 12.  13.]
    [-12. -13.]]
    >>> print(mem)
    [0 0 0 1 1 1]
    >>> print(cost)
    1.981515079547317
    >>> from sklearn.metrics import adjusted_rand_score as ARI
    >>> print('ARI =', ARI(y, mem))
    ARI = 1.0
    '''

    min_cost = +np.inf
    for _ in range(n_init):
        centers = X[np.random.choice(
            X.shape[0], size=n_clusters, replace=False
        )]
        for v_iter in range(max_iter):
            # Compute kernel similarities
            K = np.exp(
                -cdist(centers, X, metric='sqeuclidean') / (2 * (sigma ** 2))
            )
            K_dist = np.fmax(1 - K, np.finfo(np.float).eps)
            # Update memberships
            U = np.fmax(
                K_dist ** (-p / (2 * (m - 1))), np.finfo(np.float).eps
            )
            U = U / U.sum(axis=0)
            # Update centers
            old_centers = np.array(centers)
            expr_part = np.fmax((
                (U ** m) * (K_dist ** ((p - 2) / 2)) * K
            ), np.finfo(np.float).eps)
            centers = expr_part.dot(X) / expr_part.sum(axis=1)[:, None]

            if ((centers - old_centers) ** 2).sum() < tol:
                break

        # Compute cost
        cost = ((U ** m) * (K_dist ** (p / 2))).sum()
        if cost < min_cost:
            min_cost = cost
            mincost_centers = np.array(centers)
            mincost_mem = U.argmax(axis=0)
    return mincost_centers, mincost_mem, min_cost


if __name__ == '__main__':
    # DEBUG

    ''' Test1: 3 clusters
    X = np.vstack((np.vstack((
        np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
        np.random.normal(loc=[6, 0], scale=1, size=(100, 2)))),
        np.random.normal(loc=[3, 5], scale=1, size=(100, 2))))
    y = np.hstack((np.zeros((100)), np.ones((100)), np.zeros((100)) + 2))
    centers, mem, cost = kernel_gfcm(X, n_clusters=3)
    print(centers)
    from sklearn.metrics import adjusted_rand_score as ARI
    print('ARI =', ARI(y, mem))
    import matplotlib.pyplot as plt
    for iter1 in range(3):
        plt.scatter(X[mem == iter1, 0], X[mem == iter1, 1], marker='x')
    plt.show()
    '''

    ''' Test2: Iris
    from sklearn.datasets import load_iris
    X = load_iris().data
    k = 3
    centers, mem, cost = kernel_gfcm(X, n_clusters=k)
    from sklearn.metrics import adjusted_rand_score as ARI
    print('ARI =', ARI(load_iris().target, mem))
    import matplotlib.pyplot as plt
    for iter1 in range(k):
        plt.scatter(X[mem == iter1, 2], X[mem == iter1, 3], marker='x')
    plt.show()
    '''
