from ImageTransformFunctions import *

import torch
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DataGenerator(Dataset):
    """Papirus dataset."""
    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        image = ImageTransformFunctions.load_image(img_name)
        if self.training:
            mask = ImageTransformFunctions.get_mask(image, labels)
        shape = np.array(image.shape)[:2]

        image = ImageTransformFunctions.to_square(image)
        image = ImageTransformFunctions.preprocess(image, shape[0], shape[1])
        if self.training:
            mask = ImageTransformFunctions.to_square(mask)

        image = np.rollaxis(image, 2, 0)
        if self.training:
            mask = np.rollaxis(mask, 2, 0)
        else:
            mask = 0

        sample = [image, mask, shape]

        if self.transform:
            sample = self.transform(sample)

        return sample