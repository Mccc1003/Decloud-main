import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class HazyGT_Dataset(Dataset):
    def __init__(self, hazy_dir, gt_dir=None, transform=None):
        super().__init__()
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.file_names = sorted(os.listdir(hazy_dir))

        if gt_dir is not None:
            gt_files = set(os.listdir(gt_dir))
            self.file_names = [f for f in self.file_names if f in gt_files]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.file_names[idx])
        hazy_img = Image.open(hazy_path).convert('RGB')

        if self.transform:
            hazy_img = self.transform(hazy_img)

        if self.gt_dir is not None:
            gt_path = os.path.join(self.gt_dir, self.file_names[idx])
            gt_img = Image.open(gt_path).convert('RGB')
            if self.transform:
                gt_img = self.transform(gt_img)
            return hazy_img, gt_img
        else:
            return hazy_img
