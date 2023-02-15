import os
import numpy as np
from framework.utils import download
from .classification_dataset import ImageClassificationDataset, ImageSet
from .reconstruction_dataset import ImageReconstructionDataset
from typing import List, Optional, Callable
import tarfile
from .image_folder import FolderImageSet


class Imagenet(ImageClassificationDataset):
    SETS = {
        "train": 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
        # "train": 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
        "test": 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar'
    }

    DEVKIT = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"

    n_classes = 1000

    def load_mean_std(self, cache: str):
        self.mean_tensor = np.asfarray([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32) * 255.0
        self.std_tensor = np.asfarray([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32) * 255.0

    def download(self, cache: str):
        for n, url in self.SETS.items():
            print("Downloading", url)
            download(url, cache + "/" + n, ignore_if_exists=False, extract=True)

        train_dir = os.path.join(cache, "train")
        for tfile in os.listdir(train_dir):
            if not tfile.lower().endswith(".tar"):
                continue

            print(f"Extracting {tfile}...")
            full_fname = os.path.join(train_dir, tfile)
            file = tarfile.open(full_fname)
            file.extractall(train_dir)
            file.close()

            os.remove(full_fname)

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        return FolderImageSet(cache_dir + "/" + set, self.transform)

    def __init__(self, set: str, cache: str = "./cache", valid_split_size: float = 0.01,
                 normalize: bool = True, restrict: Optional[List[int]] = None,
                 augment: Callable[[np.ndarray], np.ndarray] = lambda x: x,
                 transform: Optional[Callable] = None):

        self.transform = transform
        super().__init__(
            set=set, cache=cache, valid_split_size=valid_split_size, normalize=normalize, restrict=restrict,
            augment=augment)


class ImagenetReconstruciton(ImageReconstructionDataset, Imagenet):
    pass
