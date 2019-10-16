'''Kernel k-Means Clustering'''

# Author: Avisek Gupta


import numpy as np
from scipy.spatial.distance import cdist


def kernel_kmeans(X, n_clusters, sigma=1, max_iter=300, n_init=30, tol=1e-16):
    '''Kernel k-Means Clustering

    Parameters
    ----------

    X : array, shape (n_data_points, n_features)
        The data array.

    n_clusters : int
        The number of clusters.

    sigma : float
        Parameter for the Gaussian Kernel.
        K(a,b) = np.exp(-(a-b)**2 / (2 * (sigma ** 2)))

    max_iter: int, default: 300
        The maximum number of iterations of the Kernel k-Means algorithm.

    n_init: int, default 30
        The number of runs of the Kernel k-Means algorithm.
        The results corresponding to the minimum cost will be returned.

    tol: float, default 1e-16
        Tolerance for convergence.

    Returns
    -------

    mincost_mem: array, shape (n_data_points)
        The resulting cluster memberships.

    min_cost: float
        The lowest cost achieved in n_init runs.
    '''

    K = np.exp(-cdist(X, X, metric='sqeuclidean') / (2 * (sigma ** 2)))
    K_diag = np.diag(K)
    min_cost = +np.inf
    for _ in range(n_init):
        cost = +np.inf
        mem = np.random.randint(0, n_clusters, X.shape[0])
        for v_iter in range(max_iter):
            # Compute distances
            dist = np.zeros((X.shape[0], n_clusters))
            for j in range(n_clusters):
                dist[:, j] = K_diag
                if (mem == j).sum() > 0:
                    dist[:, j] = (
                        dist[:, j]
                        - (2 * K[:, mem == j].sum(axis=1) / (mem == j).sum())
                        + (K[:, mem == j][mem == j, :].sum()
                            / ((mem == j).sum() ** 2))
                    )
            # Update membership
            mem = dist.argmin(axis=1)
            # Check for convergence
            prev_cost = cost
            cost = dist[np.arange(X.shape[0]), mem].sum()
            if prev_cost - cost < tol:
                break
        if min_cost > cost:
            min_cost = cost
            mincost_mem = np.array(mem)
    return mincost_mem, min_cost


if __name__ == '__main__':
    # DEBUG

    ''' Test1: 3 clusters
    X = np.vstack((np.vstack((
        np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
        np.random.normal(loc=[6, 0], scale=1, size=(100, 2)))),
        np.random.normal(loc=[3, 5], scale=1, size=(100, 2))))
    y = np.hstack((np.zeros((100)), np.ones((100)), np.zeros((100)) + 2))
    mem, cost = kernel_kmeans(X, n_clusters=3)
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
    mem, cost = kernel_kmeans(X, n_clusters=k)
    from sklearn.metrics import adjusted_rand_score as ARI
    print('ARI =', ARI(load_iris().target, mem))
    import matplotlib.pyplot as plt
    for iter1 in range(k):
        plt.scatter(X[mem == iter1, 2], X[mem == iter1, 3], marker='x')
    plt.show()
    '''
