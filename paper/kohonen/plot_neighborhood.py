import sys
sys.path.append('../..')

import lib

import layers
import torch

from layers.som_vector_quantizer import SOMGeometry, cos_distance, InverseUpdateSOM, l2_distance, DiffSOM, GaussianNeighborhood, Grid, HardNeighborhood
import numpy as np

import matplotlib.pyplot as plt


def plot_geometry(geom: SOMGeometry, step: int = 0):
    n = np.prod(geom.grid.shape)
    geom.init(n)
    i = torch.zeros([1,n])

    c = torch.zeros([1,2], dtype=torch.long)
    c[0] = torch.tensor([10, 10], dtype=torch.long)
    c2 = geom.grid.from_semantic_space(c)

    i[0, c2[0]] = 1

    bm = geom.blur.get_blur_matrix(torch.tensor([step]))
    bm.fill_diagonal_(1.0)
    i = i @ bm

    ind = geom.grid.to_semantic_space(torch.arange(n, dtype=torch.long))
    gridplot = torch.zeros(geom.grid.shape, dtype=torch.float32)

    for j in range(n):
        jj = ind[j]
        gridplot[jj[0],jj[1]] = i[0, j]


    fig = plt.figure()
    plt.imshow(gridplot, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks([],[])
    plt.yticks([],[])
    plt.scatter([c[0,0]], [c[0,1]], marker="+", color="red")
    return fig




g = SOMGeometry(
    Grid(2, [32,32]),
    # HardNeighborhood(0.1)
    GaussianNeighborhood(0.1, base=10)
)
plot_geometry(g, 0).savefig("neighborhood_early.pdf", bbox_inches='tight', pad_inches = 0.01)
plot_geometry(g, 100).savefig("neighborhood_late.pdf", bbox_inches='tight', pad_inches = 0.01)