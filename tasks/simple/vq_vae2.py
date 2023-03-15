import torch

import dataset
from .. import task, args

from .vq_vae import VQVae
from models.vq_vae2 import VQVAE2

from torchvision import transforms
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-vq_vae.image_size", default=256)


@task(name="vq_vae2")
class VQVae2(VQVae):
    TRAIN_NUM_WORKERS = 8
    VALID_NUM_WORKERS = 2

    def create_datasets(self):
        self.batch_dim = 0

        s = self.helper.args.vq_vae.image_size
        self.train_set = dataset.image.ImagenetReconstruciton("train", transform=transforms.Resize([s, s]))
        self.valid_sets.valid = dataset.image.ImagenetReconstruciton("test", transform=transforms.Resize([s, s]))

    def create_model(self) -> torch.nn.Module:
        geometry = self.create_geometry()

        return VQVAE2(
            n_res_channel=self.helper.args.vq_vae.num_residual_hiddens,
            channel=self.helper.args.vq_vae.num_hiddens,
            n_res_block=self.helper.args.vq_vae.num_residual_layers,
            n_embed=self.helper.args.vq_vae.num_embeddings,
            embed_dim=self.helper.args.vq_vae.embedding_dim,
            decay=self.helper.args.vq_vae.decay,
            commitment_cost=self.helper.args.vq_vae.commitment_cost,
            quantizer=self.helper.args.vq_vae.quantizer,
            som_geometry=geometry,
            counter_init=self.helper.args.vq_vae.magic_counter_init,
            som_cost=self.helper.args.vq_vae.som_cost,
        )
