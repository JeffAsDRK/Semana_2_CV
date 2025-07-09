from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image 
import torch 
from glob import glob 

DATASET_PATH = "./flood_area_dataset"
DEFAULT_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

class FloodDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=DEFAULT_TRANSFORM) -> None:
        self.img_names = []
        self.mask_names = []

        with open(f"{dataset_path}/metadata.csv", 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.strip()
                img_name, mask_name = line.split(",")
                img_string = f"./{dataset_path}/Image/{img_name}"
                mask_string = f"./{dataset_path}/Mask/{mask_name}"
                self.img_names.append(img_string)
                self.mask_names.append(mask_string)

        self.transform = transform 

    def __len__(self):
        return len(self.img_names)
    

    def __getitem__(self, idx):
    
        img_string = self.img_names[idx]
        mask_string = self.mask_names[idx]

        img = Image.open(img_string).convert("RGB")
        img.load()
        img = self.transform(img)

        mask = Image.open(mask_string).convert("RGB")
        mask.load()
        mask = self.transform(mask)
        mask = v2.Grayscale(1)(mask) # para funcionar con el weights.transforms()

        return img, mask 
    
#def load_data(dataset_path=DATASET_PATH, transform=DEFAULT_TRANSFORM, num_workers=0, batch_size=128):
#    dataset = FloodDataset(dataset_path, transform=transform)
#    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
#                      shuffle=True, drop_last=False)


class Dataset(Dataset):
    def __init__(self, df, transform=DEFAULT_TRANSFORM) -> None:
        
        self.df_class=df

        #with open(f"{dataset_path}/metadata.csv", 'r', encoding='utf-8') as f:
        #    next(f)
        #    for line in f:
        #        line = line.strip()
        #        img_name, mask_name = line.split(",")
        #        img_string = f"./{dataset_path}/Image/{img_name}"
        #        mask_string = f"./{dataset_path}/Mask/{mask_name}"
        #        self.img_names.append(img_string)
        #        self.mask_names.append(mask_string)


        self.transform = transform 

    def __len__(self):
        return len(self.df_class)
    

    def __getitem__(self, idx):
        img_string = f"dense_data/{self.df_class['name'].iloc[idx]}/frame/{self.df_class['frame'].iloc[idx]}"
        mask_string = f"dense_data/{self.df_class['name'].iloc[idx]}/combined/{self.df_class['combined'].iloc[idx]}"

        img = Image.open(img_string).convert("RGB")
        img.load()
        img = self.transform(img)
        img = v2.ToDtype(torch.float32, scale=True)(img)

        mask = Image.open(mask_string).convert("L")
        mask.load()
        mask = self.transform(mask)
        #mask = v2.Grayscale(1)(mask) # para funcionar con el weights.transforms()

        return img, mask 
    
#def load_data(dataset_path=DATASET_PATH, transform=DEFAULT_TRANSFORM, num_workers=0, batch_size=128):
#    dataset = FloodDataset(dataset_path, transform=transform)
#    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
#                      shuffle=True, drop_last=False)

def load_data(df, transform=DEFAULT_TRANSFORM, num_workers=0, batch_size=128):
    dataset = Dataset(df, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,
                      shuffle=True, drop_last=False)