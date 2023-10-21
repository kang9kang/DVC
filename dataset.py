import os
import torch
from torch.utils.data import Dataset
import imageio


class ImageProcessor:
    def __init__(self) -> None:
        pass

    def __new__(self, *args, **kwargs):
        if not hasattr(self, "_instance"):
            self._instance = super(ImageProcessor, self).__new__(self)
        return self._instance

    def random_flip(self, img_input, img_ref):
        transform = True
        horizontal_flip = True
        vertical_flip = True

        if transform and horizontal_flip:
            if torch.rand(1) > 0.5:
                img_input = torch.flip(img_input, dims=[-1])
                img_ref = torch.flip(img_ref, dims=[-1])

        if transform and vertical_flip:
            if torch.rand(1) > 0.5:
                img_input = torch.flip(img_input, dims=[-2])
                img_ref = torch.flip(img_ref, dims=[-2])

        return img_input, img_ref

    def random_crop(self, img_input, img_ref, crop_size):
        img_height, img_width = img_input.shape[-2:]
        crop_height, crop_width = crop_size

        if img_height == crop_height:
            top = 0
        else:
            top = torch.randint(low=0, high=img_height - crop_height, size=(1,)).item()
        if img_width == crop_width:
            left = 0
        else:
            left = torch.randint(low=0, high=img_width - crop_width, size=(1,)).item()

        img_input = img_input[..., top : top + crop_height, left : left + crop_width]
        img_ref = img_ref[..., top : top + crop_height, left : left + crop_width]

        return img_input, img_ref


class VideoDataset(Dataset):
    def __init__(
        self, path="data/vimeo_septuplet/train.txt", img_height=256, img_width=256
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.img_input_list, self.img_ref_list = self.get_vimeo(filelist=path)
        self.img_processor = ImageProcessor()

    def get_vimeo(
        self,
        root="data/vimeo_septuplet/sequences/",
        filelist="data/vimeo_septuplet/train.txt",
    ):
        with open(filelist, "r") as f:
            lines = f.readlines()
        img_input_list = []
        img_ref_list = []

        for line in lines:
            line = line.rstrip()
            img_input_list.append(os.path.join(root, line))
            refnum = int(line[-5:-4]) - 2
            refname = line[:-5] + str(refnum) + line[-4:]
            img_ref_list.append(os.path.join(root, refname))
        return img_input_list, img_ref_list

    def __len__(self):
        return len(self.img_input_list)

    def __getitem__(self, idx):
        img_input = imageio.imread(self.img_input_list[idx])
        img_ref = imageio.imread(self.img_ref_list[idx])

        img_input = img_input.transpose(2, 0, 1)
        img_ref = img_ref.transpose(2, 0, 1)

        img_input = img_input.astype("float32") / 255.0
        img_ref = img_ref.astype("float32") / 255.0

        img_input = torch.from_numpy(img_input)
        img_ref = torch.from_numpy(img_ref)

        img_input, img_ref = self.img_processor.random_crop(
            img_input, img_ref, [self.img_height, self.img_width]
        )
        img_input, img_ref = self.img_processor.random_flip(img_input, img_ref)

        return img_input, img_ref
