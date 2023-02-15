import os
import numpy as np
from .classification_dataset import ImageSet
from typing import List, Optional, Callable, Tuple
from PIL import Image
from .classification_dataset import ImageClassificationDataset
from .reconstruction_dataset import ImageReconstructionDataset
from framework.utils import download, LockFile
import multiprocessing
from tqdm import tqdm

from torchvision import transforms
import pathlib


class ImageLoader:
    def __init__(self, base_dir: str, file_list: List[str], transform: Optional[Callable]) -> None:
        self.file_list = file_list
        self.base_dir = base_dir
        self.transform = transform if transform is not None else lambda x: x
        self.mean = 0
        self.std = 1

    def load_image(self, item:int):
        name = os.path.join(self.base_dir, self.file_list[item])
        return Image.open(name).convert("RGB")

    def get_unnormalized_image(self, item, dtype=np.uint8):
        img = self.load_image(item)
        img = self.transform(img)
        if isinstance(img, Image.Image) or img.dtype != dtype:
            img = np.array(img, dtype=dtype)
        return img

    def __getitem__(self, item: int):
        img = self.get_unnormalized_image(item, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = (img - self.mean) / self.std
        return img

    def filter(self, mask: np.ndarray):
        selected_file_list = []
        for img, present in zip(self.file_list, mask):
            if present:
                selected_file_list.append(img)
        res = ImageLoader(self.base_dir, selected_file_list, self.transform)
        res.mean = self.mean
        res.std = self.std
        return res

    def normalize(self, mean: np.ndarray, std: np.ndarray):
        res = ImageLoader(self.base_dir, self.file_list, self.transform)
        res.mean = mean
        res.std = std
        return res

    def __len__(self):
        return len(self.file_list)


class CachedImageLoader(ImageLoader):
    def __init__(self, fname: Optional[str], shape: List[int], transform: Optional[Callable] = None,
                 index_list: Optional[List[int]] = None):
        self.transform = transform if transform is not None else lambda x: x
        self.mean = 0
        self.std = 1
        self.shape = shape
        self.index_list = index_list

        if fname is not None:
            self.cache = np.memmap(fname, dtype=np.uint8, mode="r")
            self.cache = np.reshape(self.cache, [-1, *shape, 3])
            self.index_list = self.index_list if self.index_list is not None else list(range(self.cache.shape[0]))

    def __len__(self):
        return len(self.index_list)

    def load_image(self, item: int):
        return self.cache[item]

    def filter(self, mask: np.ndarray):
        index_list = []
        for i, present in zip(self.index_list, mask):
            if present:
                index_list.append(i)

        res = CachedImageLoader(None, self.shape, self.transform, index_list)
        res.cache = self.cache
        res.mean = self.mean
        res.std = self.std
        return res

    def normalize(self, mean: np.ndarray, std: np.ndarray):
        res = CachedImageLoader(None, self.shape, self.transform, self.index_list)
        res.cache = self.cache
        res.mean = mean
        res.std = std
        return res

    @staticmethod
    def from_image_loader(dest: str, loader: ImageLoader):
        shape = loader.get_unnormalized_image(0).shape
        cache = np.memmap(dest, dtype=np.uint8, mode="w+", shape=(len(loader), *shape))

        print("Building image cache...")
        chunk_size = multiprocessing.cpu_count() * 4

        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            for i in tqdm(range(0, len(loader), chunk_size)):
                indices = list(range(i, min(i + chunk_size, len(loader))))
                images = p.map(loader.get_unnormalized_image, indices)
                for j, img in enumerate(images):
                    cache[i + j] = img

        cache.flush()
        del cache


class FolderImageSet(ImageSet):
    def __init__(self, dir: Optional[str], transform: Optional[Callable] = None,
                 resize: Optional[Tuple[int, int]] = None, cache_path: Optional[str] = None):
        if dir is None:
            return

        if cache_path is not None:
            if resize is None:
                raise ValueError("Caching is possible only if reshaping to a fixed size.")

            cache_array = os.path.join(cache_path, f"{resize[0]}_{resize[1]}.numpy")
            done_file = os.path.join(cache_path, f"{resize[0]}_{resize[1]}.done")
            lock_file = os.path.join(cache_path, "lock")

            with LockFile(lock_file):
                if not os.path.isfile(done_file):
                    if os.path.isfile(cache_path):
                        os.remove(cache_path)

                    loader = self.create_loader(dir, transforms.Resize(resize))
                    CachedImageLoader.from_image_loader(cache_array, loader)
                    pathlib.Path(done_file).touch()

            self.images = CachedImageLoader(cache_array, resize, transform)
        else:
            if transform is not None and resize is not None:
                transform = transforms.Compose([
                    transforms.Resize(resize),
                    transform
                ])
            self.images = self.create_loader(dir, transform)

        self.labels = np.zeros([len(self.images)], dtype=np.int16)

    def create_loader(self, dir: str, transform: Optional[Callable]):
        flist = []
        for root, _, files in os.walk(dir):
            root = os.path.relpath(root, dir)
            for x in files:
                xl = x.lower()
                if any([xl.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]]):
                    flist.append(os.path.join(root, x))

        flist.sort()

        return ImageLoader(dir, flist, transform)

    def filter(self, mask: np.ndarray):
        res = FolderImageSet(None)
        res.images = self.images.filter(mask)
        res.labels = self.labels[mask]
        return res

    def __len__(self):
        return self.labels.shape[0]

    def filter_by_classes(self, classes: List[int]):
        raise NotImplementedError

    def normalize(self, mean: np.ndarray, std: np.ndarray):
        res = FolderImageSet(None)
        res.images = self.images.normalize(mean, std)
        res.labels = self.labels
        return res


class SimpleImageReconstructionDataset(ImageReconstructionDataset, ImageClassificationDataset):
    URL = None
    n_classes = 2
    SPLIT_TO_FOLDER = None

    def load_mean_std(self, cache: str):
        self.mean_tensor = np.asfarray([[[0.5]], [[0.5]], [[0.5]]], dtype=np.float32) * 255.0
        self.std_tensor = np.asfarray([[[0.5]], [[0.5]], [[0.5]]], dtype=np.float32) * 255.0

    def download(self, cache: str):
        print("Downloading", self.URL)
        download(self.URL, cache + "/", ignore_if_exists=False, extract=True)

    def load_data(self, cache_dir: str, set: str) -> ImageSet:
        if self.cache_loads:
            cache_name = os.path.join(cache_dir, "caches", set)
            os.makedirs(cache_name, exist_ok=True)
        else:
            cache_name = None

        return FolderImageSet(
            os.path.join(cache_dir, self.SPLIT_TO_FOLDER[set]), self.transform, self.resize, cache_name)

    def __init__(self, set: str, cache: str = "./cache", valid_split_size: float = 0.05,
                 normalize: bool = True, restrict: Optional[List[int]] = None,
                 augment: Callable[[np.ndarray], np.ndarray] = lambda x: x,
                 transform: Optional[Callable] = None, resize: Optional[Tuple[int, int]] = None,
                 cache_loads: bool = False):

        self.resize = resize
        self.transform = transform
        self.cache_loads = cache_loads
        super().__init__(
            set=set, cache=cache, valid_split_size=valid_split_size, normalize=normalize, restrict=restrict,
            augment=augment)
