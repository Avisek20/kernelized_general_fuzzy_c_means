'''Generate BIRCH-style clusters, with differing densities.
Data clusters are generated on 4x4, 5x5, 6x6, 7x7 grids.
Saved in clusters4, clusters5, clusters6, clusters7 folders.
1000 data sets generated for each grid.
'''

# Author: Avisek Gupta


import os
import numpy as np

num_datasets = 1000

for grid_size in range(4, 7+1):
    # Number of clusters
    num_clusters = grid_size * grid_size
    # Cluster locations on a grid
    list_indices = np.arange(num_clusters)

    if not os.path.exists('clusters'+str(grid_size)):
        os.makedirs('clusters'+str(grid_size))

    for iter0 in range(num_datasets):
        dataset = []
        for iter1 in range(num_clusters):
            # Set Mean
            mean = np.array([
                3 * int(list_indices[iter1] / grid_size),
                3 * int(list_indices[iter1] % grid_size)
            ])
            # Set Covariance Matrix
            cov = np.array([[0.5, 0], [0, 0.5]])

            # Randomly set density
            density = np.random.rand() > 0.5
            if density == 1:
                num_points = np.random.randint(200, 250+1)
            else:
                num_points = np.random.randint(40, 50+1)

            sample = np.random.multivariate_normal(mean, cov, num_points)

            # Add sample to dataset
            if iter1 == 0:
                dataset = np.hstack((
                    sample, np.zeros((num_points, 1))))
            else:
                dataset = np.vstack((
                    dataset,
                    np.hstack((sample, np.zeros((num_points, 1)) + iter1))
                ))

        # Save dataset
        np.savetxt(
            'clusters' + str(grid_size) + '/dataset' + str(iter0+1), dataset
        )
