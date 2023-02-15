from .image_folder import SimpleImageReconstructionDataset


class AFHQ(SimpleImageReconstructionDataset):
    URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"
    SPLIT_TO_FOLDER = {
        "train": "afhq/train",
        "test": "afhq/val"
    }

class AFHD2(AFHQ):
    URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
