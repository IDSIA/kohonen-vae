import dataset
from .simple_task import SimpleTask
import framework
import torch
from .. import task, args
from models.vq_vae import VQVAE, SOMGeometry, HardNeighborhood, Grid, EmptyNeigborhood, GaussianNeighborhood
from interfaces import ImageReconstructionInterface


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-vq_vae.num_hiddens", default=128)
    parser.add_argument("-vq_vae.num_residual_layers", default=2)
    parser.add_argument("-vq_vae.num_residual_hiddens", default=32)
    parser.add_argument("-vq_vae.num_embeddings", default=512)
    parser.add_argument("-vq_vae.commitment_cost", default=0.25)
    parser.add_argument("-vq_vae.decay", default=0.99)
    parser.add_argument("-vq_vae.count_unit", default=0.1)
    parser.add_argument("-vq_vae.embedding_dim", default=64)
    parser.add_argument("-vq_vae.grid_dim", default=1)
    parser.add_argument("-vq_vae.gaussaian_base", default=10.0)
    parser.add_argument("-vq_vae.neihborhood", default="hard", choice=["hard", "none", "gaussian"])
    parser.add_argument("-vq_vae.quantizer", default="hard_som",
                        choice=["hard_som", "hardsom_noupdate_zero"])
    parser.add_argument("-vq_vae.magic_counter_init", default=0.0)


def create_geometry(args) -> SOMGeometry:
    if args.vq_vae.neihborhood == "hard":
        neighborhood = HardNeighborhood(args.vq_vae.count_unit)
    elif args.vq_vae.neihborhood == "none":
        neighborhood = EmptyNeigborhood()
    elif args.vq_vae.neihborhood == "gaussian":
        neighborhood = GaussianNeighborhood(args.vq_vae.count_unit,
                                            base=args.vq_vae.gaussaian_base)
    else:
        raise ValueError(f"Invalid neighborhood: {args.vq_vae.neihborhood}")

    return SOMGeometry(
        Grid(args.vq_vae.grid_dim),
        neighborhood
    )


@task(name="vq_vae")
class VQVae(SimpleTask):
    VALID_NUM_WORKERS = 0

    def create_datasets(self):
        self.batch_dim = 0

        self.train_set = dataset.image.CIFAR10Reconstruction("train")
        self.valid_sets.valid = dataset.image.CIFAR10Reconstruction("valid")

    def create_geometry(self) -> SOMGeometry:
        return create_geometry(self.helper.args)

    def create_model(self) -> torch.nn.Module:
        geometry = self.create_geometry()

        return VQVAE(
            self.helper.args.vq_vae.num_hiddens, self.helper.args.vq_vae.num_residual_layers,
            self.helper.args.vq_vae.num_residual_hiddens, self.helper.args.vq_vae.num_embeddings,
            self.helper.args.vq_vae.embedding_dim, self.helper.args.vq_vae.commitment_cost,
            decay=self.helper.args.vq_vae.decay,
            quantizer=self.helper.args.vq_vae.quantizer,
            som_geometry=geometry, magic_counter_init=self.helper.args.vq_vae.magic_counter_init)

    def create_model_interface(self):
        self.model_interface = ImageReconstructionInterface(self.model)
