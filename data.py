import cv2
import torch
from torch.utils.data import Dataset,DataLoader
class SorghumDataset(Dataset):
        def __init__(self, df, transform=None):
            self.image_path = df['file_path'].values
            self.labels = df["cultivar_index"].values
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
    #         image_id = self.image_id[idx]
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            image_path = self.image_path[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = self.transform(image=image)
            image = augmented['image']
            return {'image':image,
                    'target': label}
        
        
def get_loader(df, transform, **loader_args):
    dset = SorghumDataset(df, transform)
    loader = DataLoader(dset, **loader_args, persistent_workers=False)
    return dset, loader
