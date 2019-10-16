# Comparison of clustering algorithms

This folder contains the files to conduct the experiment (reported in the source paper) comparing the performance of Kernel k-Harmonic Means with Kernel Fuzzy c-Means, Kernel k-Means, and BIRCH.

Explanation for each of the files are provided below -

* **kernel_gfcm.py**: Source code of Kernelized General Fuzzy c-Means.
* **kernel_kmeans.py**: Source code of Kernel k-Means.
* **gen_largenum_clusters1.py**: Code to generate BIRCH-like clusters on a grid of size 4x4, 5x5, 6x6 and 7x7. Clusters are randomly set to be dense or sparse.
* **plot_data.py**: Helper code to visualize any of the generated data.
* **cluster_kkhm1.py**: Code to compare the clustering performance of Kernel k-Harmonic Means with Kernel Fuzzy c-Means, Kernel k-Means, and BIRCH, on the data generated on grids of size 4x4, 5x5, 6x6 or 7x7.
