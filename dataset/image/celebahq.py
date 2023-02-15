from .image_folder import SimpleImageReconstructionDataset


class CelebaHQ(SimpleImageReconstructionDataset):
    URL = "https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=1"
    n_classes = 2
    SPLIT_TO_FOLDER = {
        "train": "celeba_hq/train",
        "test": "celeba_hq/val"
    }
