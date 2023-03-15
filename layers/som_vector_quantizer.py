# Code based on https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py and
# https://github.com/zalandoresearch/pytorch-vq-vae

from typing import Union, List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from dataclasses import dataclass
import torch.distributed


class Grid:
    Shape = Union[List[int], int]

    def __init__(self, dim: int, shape: Optional[Shape] = None):
        self.dim = dim
        self.shape = shape

    def decompose_factors(self, n: int, dim: int) -> List[int]:
        if dim == 1:
            return [n]

        h = math.ceil(n ** (1 / dim))
        for i in range(h, 0, -1):
            if n % i == 0:
                w = n // i
                return self.decompose_factors(w, dim - 1) + [i]

        raise ValueError("Factorization failed.")

    def init(self, n_total: int):
        if self.shape is None:
            self.shape = self.decompose_factors(n_total, self.dim)
            print(f"Warning: Grid size auto-set to {self.shape}")

        self.shape = self.shape if isinstance(self.shape, list) else [self.shape] * self.dim
        self.n_elem = np.prod(self.shape)

        if self.n_elem != n_total:
            raise ValueError("The grid shape doesn't match the total number of elements.")

        self._id_to_coord = torch.empty([self.n_elem, self.dim], dtype=torch.int32)
        self._coord_to_id = {}

        for i in range(self.n_elem):
            coord = []
            i_rem = i
            for s in self.shape:
                coord.append(i_rem % s)
                i_rem //= s

            self._id_to_coord[i] = torch.tensor(coord, dtype=torch.int32)
            self._coord_to_id[tuple(coord)] = i

        cf = self._id_to_coord.float()
        self._distances = (cf.unsqueeze(0) - cf.unsqueeze(1)).norm(dim=-1)

    def pairwise_distance(self) -> torch.Tensor:
        return self._distances

    def __getitem__(self, addr: Union[int, Sequence[int]]) -> Union[int, Sequence[int]]:
        if isinstance(addr, int):
            return self._id_to_coord[addr]
        else:
            return self._coord_to_id[tuple(addr)]

    def to_semantic_space(self, t: torch.Tensor) -> torch.Tensor:
        return self._id_to_coord.to(t.device)[t].type_as(t)

    def from_semantic_space(self, t: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(t.shape[:-1], dtype=t.dtype, device=t.device)
        for i in range(len(self.shape)):
            # coord.append(i_rem % s)
            res += t[..., i]
            t = t * self.shape[i]
        return res

    def clamp_to_limits(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        for i, s in enumerate(self.shape):
            t[..., i][t[..., i] < 0] = 0
            t[..., i][t[..., i] >= s] = s - 1
        return t


class SOMBlur:
    def set_distances(self, distance_matrix: torch.Tensor):
        raise NotImplementedError

    def get_blur_matrix(self, step: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HardNeighborhood(SOMBlur):
    def __init__(self, shrink_step: float) -> None:
        self.shrink_step = shrink_step

    def set_distances(self, distance_matrix: torch.Tensor):
        self.neighborhood = (distance_matrix < 1.5).float()

    def get_blur_matrix(self, step: torch.Tensor) -> torch.Tensor:
        if self.neighborhood.device != step.device:
            self.neighborhood = self.neighborhood.to(step.device)
        return self.neighborhood / (1. + step * self.shrink_step)


class EmptyNeigborhood(SOMBlur):
    def set_distances(self, distance_matrix: torch.Tensor):
        self.neighborhood = torch.zeros_like(distance_matrix)

    def get_blur_matrix(self, step: torch.Tensor) -> torch.Tensor:
        if self.neighborhood.device != step.device:
            self.neighborhood = self.neighborhood.to(step.device)
        return self.neighborhood


class GaussianNeighborhood(SOMBlur):
    def __init__(self, shrink_step: float, base: float = 100) -> None:
        self.shrink_step = shrink_step
        self.base = base

    def set_distances(self, distance_matrix: torch.Tensor):
        self.distance_matrix = distance_matrix

    def get_blur_matrix(self, step: torch.Tensor) -> torch.Tensor:
        if self.distance_matrix.device != step.device:
            self.distance_matrix = self.distance_matrix.to(step.device)

        return (self.distance_matrix * ((1 + step * self.shrink_step) / (-self.base))).exp()



@dataclass
class SOMGeometry:
    grid: Grid
    blur: SOMBlur

    def init(self, n_total: int):
        self.grid.init(n_total)
        self.blur.set_distances(self.grid.pairwise_distance())


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (torch.sum(a**2, dim=1, keepdim=True)
            + torch.sum(b**2, dim=1)
            - 2 * torch.matmul(a, b.t()))


def cos_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0-torch.matmul(a, b.t()) / (a.norm(dim=1, keepdim=True) * b.norm(dim=1))


class HardSOM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay, geometry: SOMGeometry, epsilon=1e-5,
                 magic_counter_init: float = 1.0, commitment_cost: float = 0.25):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        embed = torch.randn(self._num_embeddings, self._embedding_dim)
        self.register_buffer("_w", embed)
        self.register_buffer("_ema_w", embed)
        self.register_buffer('_ema_cluster_size', torch.full([num_embeddings], fill_value=magic_counter_init))
        self.register_buffer("counter", torch.tensor(0.))

        self._decay = decay
        self._epsilon = epsilon


        self.geometry = geometry
        self.geometry.init(self._num_embeddings)

    def update(self, encodings, flat_input):
        proj = self.geometry.blur.get_blur_matrix(self.counter)
        proj.fill_diagonal_(1.0)
        encodings_sum = encodings @ proj

        n_writes = torch.sum(encodings_sum, 0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(n_writes)

        self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * n_writes

        # Laplace smoothing of the cluster size
        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
            (self._ema_cluster_size + self._epsilon)
            / (n + self._num_embeddings * self._epsilon) * n)

        # encodings_sum = encodings
        dw = torch.matmul(encodings_sum.t(), flat_input)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(dw)

        self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw
        self._weight = self._ema_w / self._ema_cluster_size.unsqueeze(1)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        with torch.no_grad():
            # Calculate distances a^2 + b^2 - 2ab
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                        + torch.sum(self._w**2, dim=1)
                        - 2 * torch.matmul(flat_input, self._w.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self._num_embeddings,
                device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = self.embed_code(encoding_indices).view(input_shape)

            # Use EMA to update the embedding vectors
            if self.training:
                self.update(encodings, flat_input)
                self.counter += 1

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Loss
        loss = F.mse_loss(quantized.detach(), inputs) * self._commitment_cost

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encoding_indices

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self._w)


class HardSOM_noupdate_zero(HardSOM):
    def update(self, encodings, flat_input):
        proj = self.geometry.blur.get_blur_matrix(self.counter)
        proj.fill_diagonal_(1.0)
        encodings_sum = encodings @ proj

        n_writes = torch.sum(encodings_sum, 0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(n_writes)

        updated = n_writes > 0

        self._ema_cluster_size = torch.where(
            updated,
            self._ema_cluster_size * self._decay + (1 - self._decay) * n_writes,
            self._ema_cluster_size
        )

        # encodings_sum = encodings
        dw = torch.matmul(encodings_sum.t(), flat_input)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(dw)

        self._ema_w = torch.where(
            updated.unsqueeze(-1),
            self._ema_w * self._decay + (1 - self._decay) * dw,
            self._ema_w
        )

        self._w = torch.where(
            updated.unsqueeze(-1),
            self._ema_w / self._ema_cluster_size.unsqueeze(1),
            self._w
        )


class GDSOM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, geometry: SOMGeometry, commitment_cost: float,
                 kohonen_cost: float = 1.0):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._kohonen_cost = kohonen_cost

        self.w = torch.nn.Parameter(torch.randn(self._num_embeddings, self._embedding_dim))
        self.register_buffer("counter", torch.tensor(0.))

        self.geometry = geometry
        self.geometry.init(self._num_embeddings)

    def l2_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The other implementation is pairwise.
        xn = torch.norm(x, p=2, dim=-1)**2
        yn = torch.norm(y, p=2, dim=-1)**2
        return xn + yn - 2 * torch.matmul(x, y.transpose(-2,-1)).squeeze(-1)

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        with torch.no_grad():
            # Calculate distances a^2 + b^2 - 2ab
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                        + torch.sum(self.w**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.w.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self._num_embeddings,
                device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

        quantized = self.embed_code(encoding_indices).view(input_shape)

        # Kohonen loss
        diff = self.l2_batch(self.w, quantized.unsqueeze(-2).detach()).flatten(end_dim=-2)

        bmat = self.geometry.blur.get_blur_matrix(self.counter)
        selected = bmat[encoding_indices.squeeze(-1)]

        kohonen_loss = torch.bmm(diff.unsqueeze(-2), selected.unsqueeze(-1))
        kohonen_loss = kohonen_loss.mean()

        if self.training:
            self.counter += 1

        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss + self._kohonen_cost * kohonen_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encoding_indices

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.w)
