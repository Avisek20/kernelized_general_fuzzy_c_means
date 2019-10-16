'''Compare Kernel k-Harmonic Means, Kernel Fuzzy c-Means, Kernel k-Means,
and BIRCH.
Cluster data generated on 4x4, 5x5, 6x6, or 7x7 grids.
Resulting ARIs in are saved in output4, output5, output6, output7 folders.
'''

# Author: Avisek Gupta


import os
import numpy as np
from kernel_gfcm import kernel_gfcm
from kernel_kmeans import kernel_kmeans
from sklearn.cluster import Birch
from sklearn.metrics import adjusted_rand_score as ARI


# Set cluster folder number
cluster_folder_number = 4

if not os.path.exists('output' + str(cluster_folder_number)):
    os.makedirs('output' + str(cluster_folder_number))

fw = open(
    'output' + str(cluster_folder_number)
    + '/ari_output' + str(cluster_folder_number), 'a'
)

# Execute for 1000 datasets
for iter1 in range(1000):
    print('Dataset' + str(iter1 + 1))
    X = np.loadtxt(
        'clusters' + str(cluster_folder_number) + '/dataset' + str(iter1 + 1)
    )
    y = X[:, -1]
    X = X[:, 0:-1]
    k = len(np.unique(y))
    sigma = np.mean(np.var(X, axis=1))

    # Run Kernelized KHM
    kkhm_ari = -1
    for p in range(2, 3+1):
        kkhm_centers, kkhm_mem, kkhm_cost = kernel_gfcm(
            X, n_clusters=k, m=2, p=p, sigma=sigma,
            max_iter=500, n_init=5, tol=1e-6
        )
        t_ari = ARI(y, kkhm_mem)
        if t_ari > kkhm_ari:
            kkhm_ari = t_ari
    fw.write(str(kkhm_ari))
    fw.write(' ')

    # Run Kernelized FCM
    kfcm_centers, kfcm_mem, kfcm_cost = kernel_gfcm(
        X, n_clusters=k, m=2, p=2, sigma=sigma,
        max_iter=500, n_init=5, tol=1e-6
    )
    kfcm_ari = ARI(y, kfcm_mem)
    fw.write(str(kfcm_ari))
    fw.write(' ')

    # Run Kernel K-means
    kkmeans_mem, kkmeans_cost = kernel_kmeans(
        X, n_clusters=k, max_iter=500, sigma=sigma, n_init=5, tol=1e-6
    )
    kkmeans_ari = ARI(y, kkmeans_mem)
    fw.write(str(kkmeans_ari))
    fw.write(' ')

    # Run BIRCH
    brc = Birch(
        branching_factor=50, n_clusters=k,
        threshold=0.5, compute_labels=True
    )
    brc.fit(X)
    birch_mem = brc.predict(X)
    birch_ari = ARI(y, birch_mem)
    fw.write(str(birch_ari))
    fw.write('\n')

fw.close()
