import torch
from layers.som_vector_quantizer import SOMGeometry, Grid, HardSOM, HardNeighborhood

geometry = SOMGeometry(
    Grid(2),
    HardNeighborhood(0.1)
)

quantizer = HardSOM(128, 512, 0.99, geometry)

input = torch.randn(64, 512)
loss, output, perplexity, _ = quantizer(input)
