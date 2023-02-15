from typing import Dict, Any

import torch
import torch.utils.data
import numpy as np

import framework


class ImageReconstructionTest:
    def __init__(self, owner, n_vis: int = 32):
        self.owner = owner

        self.n_ok = 0
        self.n_total = 0
        self.confusion = 0
        self.n_vis = n_vis
        self.hist = []

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        for b in range(net_out.shape[0]):
            if len(self.hist) < self.n_vis or np.random.rand() < (1.0 / len(self.owner)):
                self.hist.append((net_out[b].detach().cpu(), data["image"][b].cpu(), data["label"][b].cpu()))

        if len(self.hist) > self.n_vis:
            self.hist = [self.hist[i] for i in np.random.permutation(len(self.hist))[:self.n_vis]]

    @property
    def accuracy(self) -> float:
        return 0

    def plot(self) -> Dict[str, Any]:
        data = [self.hist[i] for i in np.random.permutation(len(self.hist))][:self.n_vis]
        self.hist = []

        res = {}
        for i, d in enumerate(data):
            images = [d[0], d[1]]
            images = [self.owner.unnormalize(i).permute(1,2,0).int().clamp(0,255) for i in images]
            res[f"example_{i}"] = framework.visualize.plot.ImageGrid(images, [1, 2])

        return res


class ImageReconstructionDataset:
    def start_test(self) -> ImageReconstructionTest:
        return ImageReconstructionTest(self)
