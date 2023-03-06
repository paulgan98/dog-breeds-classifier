import torchvision.transforms as tt
from torch.utils.data import Dataset
from torch import manual_seed
manual_seed(42)

from PIL import Image, ImageEnhance
import pandas as pd
from os.path import join


class MyDataset(Dataset):
    def __init__(self, dataframes, project_dir):
        self.dataframe = pd.concat(dataframes, axis=0)
        self.project_dir = project_dir
        self.class_col = "breed"

        self.classes = sorted(list(self.dataframe[self.class_col].unique()))
        self.classes_encoded = {c : i for i, c in enumerate(self.classes)}

        self.convert_tensor = tt.ToTensor()
        self.convert_pil = tt.ToPILImage()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx, :]
        img_path = join(self.project_dir, row["imagePath"])
        image = Image.open(img_path)

        # if image has alpha channel, convert to torch tensor, slice off alpha, then convert back to PIL image
        if len(image.getbands()) == 4:
            image = self.convert_tensor(image)
            image = image[:3, :, :]
            image = self.convert_pil(image)

        label = row[self.class_col]
        label_encoded = self.classes_encoded[label]
        return image, label_encoded
    
    
class ApplyTransform(Dataset):
    def __init__(self, dataset, size, transform=None):
        self.dataset = dataset
        self.transform = transform

        # Define data augmentation transforms
        self.train_transform = tt.Compose([
            tt.Resize((size, size)),
            tt.RandomAffine(degrees=30,
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.1),
                            shear=10),
            tt.RandomHorizontalFlip(),
            tt.RandomVerticalFlip(),
            tt.ColorJitter(brightness=(0.9, 1.1),
                           hue=0.3),
            tt.ToTensor() # normalize pixel values
        ])

        self.test_transform = tt.Compose([
            tt.Resize((size, size)),
            tt.ToTensor() # normalize pixel values
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform == "train":
            image = ImageEnhance.Sharpness(image).enhance(1.2) # apply sharpening
            image = self.train_transform(image)
        elif self.transform == "test":
            image = self.test_transform(image)
    
        return image, label


def create_dataset(project_dir, csv_files):
    dfs = [pd.read_csv(f) for f in csv_files]
    ds = MyDataset(dfs, project_dir)
    return ds