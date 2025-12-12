import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil
from natsort import natsorted

from PIL import Image
import albumentations as A

# apply seeding
random.seed(0)
np.random.seed(0)



def set_dirs(datasetPath):
    all_dir = glob.glob(datasetPath + "*.png")
    images_dir = [image for image in all_dir if "label" not in image]
    labels_dir = [image.replace(".png", "_label.png") for image in images_dir] 
    return images_dir, labels_dir



def sliding_window(image, label, output_dir_images, output_dir_labels, tile_width, tile_height, sliding_size, count ):
    for y in range(0, image.shape[0] - tile_height + 1, sliding_size):
        for x in range(0, image.shape[1] - tile_width + 1, sliding_size):
            img_tile = image[y:y + tile_height, x:x + tile_width, :]
            lbl_tile = label[y:y + tile_height, x:x + tile_width]
            

            # Save tile
            img_tile_path = os.path.join(output_dir_images, f"{count}.png")
            lbl_tile_path = os.path.join(output_dir_labels, f"{count}_label.png")
            cv2.imwrite(img_tile_path, img_tile)
            cv2.imwrite(lbl_tile_path, lbl_tile)

            count += 1
    return count+1

def load_dataset(path, mode):
    images = []
    labels = []

    if mode=="saved":
        label_path = os.path.join(path, "original/binary_instance_npy/paired_images.npy")
        
        image_path = os.path.join(path, "original/tissue_images/")
        image_paths = os.listdir(image_path)
        image_paths = [image for image in image_paths if ".tif" in image]

        for image in image_paths:
            images.append(cv2.imread(os.path.join(image_path, image)))
        
        labels = np.load(label_path)

    elif mode=="mixed":
        images_dir, labels_dir = set_dirs(datsaetPath)
        print("Loading images and labels...")

        for image_dir, label_dir in tqdm(zip(images_dir, labels_dir)):
            images.append(cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE))
            labels.append(cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE))

    else:
        assert False, "Not Configured!"



    assert len(images) == len(labels), "Number of images and labels are not equal!"
    return images, labels

def re_color(label):
    unique_labels = np.unique(label)

    color_map = {label_id: [random.randint(0, 255) for _ in range(3)] for label_id in unique_labels if label_id != 0}

    colorized_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    for label_id, color in color_map.items():
        colorized_label[label == label_id] = color

    return colorized_label



if __name__ == "__main__":
    
    tileHeight = 256  # Sliding Window height
    tileWidth = 256  # Sliding Window Width
    slidingSize = 50  # Slide Skipping size
    augmentPerImage = 150  # Number of augmentations per image

    # Path to the dataset
    datsaetPath = "/mnt/vol3/Datasets/CryoNuSeg/"

    # Temporary directories for sliding window images and labels
    tmp_out = "/mnt/vol3/Datasets/CryoNuSeg/tmp_slide/"

    # Augmentation output directory
    aug_out = "/mnt/vol3/Datasets/CryoNuSeg/augmented/"

    SAVED_IMAGES = True
    DO_SLIDING_WINDOW = False
    


    if DO_SLIDING_WINDOW:
        os.makedirs(tmp_out, exist_ok=True)
        images_dir, labels_dir = load_dataset(datsaetPath, "saved")
        count = 0
        for i in tqdm(range(len(images_dir))):
            image = images_dir[i]
            label = labels_dir[i][:,:,1] # save only the instance mask
            count = sliding_window(image, label, tmp_out, tmp_out, tileWidth, tileHeight, slidingSize, count)
        print(f"Number of sliding window images: {count}")
        

    if not DO_SLIDING_WINDOW:
        os.makedirs(aug_out, exist_ok=True)
        images, labels = load_dataset(datsaetPath, "saved")

        # divide train, test
        ratio = 0.8
        train_size = int(len(images) * ratio)
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        test_images = images[train_size:]
        test_labels = labels[train_size:]

        
        # save test images
        print("Saving test images...")
        for i in tqdm(range(len(test_images))):
            image = test_images[i]
            label = test_labels[i][:,:,1]
            label = re_color(label)
            save_path = os.path.join(datsaetPath, "instanceSeg/test/")
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{i}.png"), image)
            cv2.imwrite(os.path.join(save_path, f"{i}_label.png"), label)

        
        count = 0
        print("Augmenting train images...")
        for i in tqdm(range(len(train_images))):  
            image = train_images[i]
            label = train_labels[i][:,:,1] # save only the instance mask
            label = re_color(label) # Recolor the label

            for i in range(augmentPerImage):
                aug = A.Compose([
                    A.RandomRotate90(),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomGamma(),
                    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10)),
                    A.ElasticTransform(alpha=100, sigma=50, p=0.3),
                    A.Blur(blur_limit=(3, 5), p=0.4),
                    A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.7, 1.5), hue=(0, 0),  p=0.7),
                    A.RandomCrop(height=tileHeight, width=tileWidth, p=1)

                ])
                augmented = aug(image=image, mask=label)
                aug_image = augmented['image']
                aug_label = augmented['mask']
                
                cv2.imwrite(os.path.join(aug_out, f"{count}.png"), aug_image)
                cv2.imwrite(os.path.join(aug_out, f"{count}_label.png"), aug_label)
                count += 1
            count += 1
        print(f"Number of augmented images: {count}")

        # SPLIT TRAIN, VAL  90, 10
        ratio = 0.9
        print("Splitting train, val images...")
        imagesPaths = [ imagePath for imagePath in os.listdir(aug_out) if not "label" in imagePath]
        imagesPaths = [imagePath for imagePath in imagesPaths if ".png" in imagePath]
        # randomize the images
        random.shuffle(imagesPaths)
        train_size = int(len(imagesPaths) * ratio)
        train_images = imagesPaths[:train_size]
        val_images = imagesPaths[train_size:]

        train_path = os.path.join(datsaetPath, "instanceSeg/train/")
        os.makedirs(train_path, exist_ok=True)
        val_path = os.path.join(datsaetPath, "instanceSeg/val/")
        os.makedirs(val_path, exist_ok=True)

        # copy train images
        count = 0
        for image in train_images:
            shutil.copy(os.path.join(aug_out, image), train_path+str(count)+".png")
            shutil.copy(os.path.join(aug_out, image.replace(".png", "_label.png")), train_path + str(count) + "_label.png")
            count += 1
        
        # copy val images
        count = 0
        for image in val_images:
            shutil.copy(os.path.join(aug_out, image), val_path+str(count)+".png")
            shutil.copy(os.path.join(aug_out, image.replace(".png", "_label.png")), val_path+str(count)+"_label.png")
            count += 1

        print(f"Number of train images: {len(train_images)}")
        print(f"Number of val images: {len(val_images)}")

        # remove augmented directory
        shutil.rmtree(aug_out)


        
        

