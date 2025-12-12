import glob
import os 
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset

import src.utils.conutils as utils



class nucleiSegDataset(Dataset):

    def __init__(self, config: OmegaConf, mode: str):
        super().__init__()
        self.config = config

        self.mode = mode
        self.debug = config.other.debug
        self.debug_dilution = 8 if self.debug else None
        self.segment_type = config.model.segmentation_type


        self.image_files, self.mask_files = self.setImagePaths(self.config, self.mode)
        if self.debug_dilution is not None:
            self.image_files = self.image_files[:self.debug_dilution]
            self.mask_files = self.mask_files[:self.debug_dilution]
        #print(f"Number of images in {self.mode} dataset ({self.mode}): {len(self.image_files)}")
        #print(f"Number of masks in {self.mode} dataset ({self.mode}): {len(self.mask_files)}")


    def setImagePaths(self, config: dict, mode:str) -> list:

        # check mode and set directory
        if mode == "train":
            path = os.path.join(config.dataset.path, 'train/')
        elif mode == "val":
            path = os.path.join(config.dataset.path, 'val/')
        elif mode == "test":
            path = os.path.join(config.dataset.path, 'test/')
        
        # get images and labels in the dataset
        image_files = []
        mask_files = []
        
        for file in glob.glob(os.path.join(path, "*.png")):
            if "label" not in file:
                image_files.append(file)
                mask_files.append(file.replace(".png", "_label.png"))


        # check for consistency
        '''tofile = f"../src/data/{mode}_file1.txt"
        with open(tofile, "w") as f:
            for i, m in zip(image_files, mask_files):
                f.write(f"{i},{m}\n")'''
        

        return image_files, mask_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> list:

        '''print("image: ", self.image_files[index])
        print("mask: ", self.mask_files[index])'''

        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.segment_type == "instance":
            label = cv2.imread(self.mask_files[index])
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        else:
            label = cv2.imread(self.mask_files[index], cv2.IMREAD_GRAYSCALE)
        

        #print(f"Image shape: {image.shape}, Mask shape: {label.shape}")

        if self.segment_type == "instance":
            # Generate instance IDs from color mask
            h, w = label.shape[:2]
            rgb_array = label.reshape(-1, 3)
            # Compute unique integer for each RGB color
            rgb_int = np.dot(rgb_array.astype(np.uint32), [1, 256, 256*256])
            rgb_int = rgb_int.reshape(h, w)


            # Create instance mask (background is 0, instances start from 1)
            unique_ints = np.unique(rgb_int)
            unique_ints = unique_ints[unique_ints != 0]  # Exclude background (0)
            instance_mask = np.zeros((h, w), dtype=np.int32)
            for idx, int_val in enumerate(unique_ints, start=1):
                instance_mask[rgb_int == int_val] = idx


        transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
        ])

        if self.mode =="test":
            augmented = transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        #print(f"Image shape: {image.shape}, Mask shape: {label.shape}")            

        # normalize and transform image
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        if self.segment_type == "instance":
            inputs = {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(instance_mask, dtype=torch.int64)  # Instance IDs
            }
            return inputs
        else:
            # convert mask to binary using opencv
            label = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)[1]

            # generate class mask from binary mask
            mask1 = (label == 0)
            mask2 = (label == 1)
            mask = np.stack([mask1, mask2], axis=0).astype(np.uint8)

            #print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

            inputs = {
                "image": torch.tensor(image, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.float32)
            }

            return inputs
    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)