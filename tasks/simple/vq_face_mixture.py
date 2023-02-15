
import dataset
from .. import task

from .vq_vae import VQVae
from .vq_vae2 import VQVae2

import framework


class FaceMixtureDatasetMixin:
    def create_datasets(self):
        self.batch_dim = 0

        s = self.helper.args.vq_vae.image_size

        self.train_set = framework.loader.DatasetMerger([
            dataset.image.CelebaHQ("train", resize=[s, s], cache_loads=True),
            dataset.image.AFHQ("train", resize=[s, s], cache_loads=True)
        ])

        self.valid_sets.celebahq = dataset.image.CelebaHQ("test", resize=[s, s], cache_loads=True)
        self.valid_sets.afhd = dataset.image.AFHQ("test", resize=[s, s], cache_loads=True)


@task()
class VQVae2FaceMixture(FaceMixtureDatasetMixin, VQVae2):
    pass


@task()
class VQVaeFaceMixture(FaceMixtureDatasetMixin, VQVae):
    pass
