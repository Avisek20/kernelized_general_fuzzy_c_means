# Kernelized General Fuzzy c-Means Clustering

Paper Source: [Gupta A., Das S., ``On the Unification of k-Harmonic Means and Fuzzy c-Means Clustering Problems under Kernelization'', *in the 2017 Ninth International Conference on Advances in Pattern Recognition (ICAPR 2017)*, pp. 386-391, 2017.](https://ieeexplore.ieee.org/iel7/8577948/8592935/08593078.pdf)

## Example of usage

```
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
```
