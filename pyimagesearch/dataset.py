# import the necessary packages
import skimage
import torch
from pyimagesearch import config

from torch.utils.data import Dataset
from torchvision import transforms as torch_transforms
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, Resize, ToTensor
from skimage import io
# import cv2

class CustomTransform:
    class CustomTransform:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, mask):
            for transform in self.transforms:
                image = transform(image)
                mask = transform(mask)
            return image, mask

        # Apply each transform to image and mask
            for transform in self.transforms.transforms:
                image_pil = transform(image_pil)
                mask_pil = transform(mask_pil)

        # Convert PIL Images back to float32
            image = np.array(image_pil) / 255.0
            mask = np.array(mask_pil)

            return image, mask


class SegmentationDataset(Dataset):
    
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        print(len(imagePaths), len(maskPaths))

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
                # convert to float32 for normalization
        image = skimage.io.imread(imagePath).astype('float32')

        # select bands 7, 5, and 4 (indexing starts from 0)
        swir = image[:, :, 6]  # Band 7
        nir = image[:, :, 4]   # Band 5
        red = image[:, :, 3]   # Band 4

        # stack the selected bands to create the SWIR and Natural Color Composite
        swir_natural_color = np.stack([swir, nir, red], axis=-1)

        # normalize the image to the range [0, 1]
        swir_natural_color /= 65535.0

        # read the associated mask from disk using skimage.io
        mask = skimage.io.imread(self.maskPaths[idx]).astype('float32')

        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        # image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.maskPaths[idx], 0)
        # check to see if we are applying any transformations

        
        if self.transforms is not None:
        # Convert NumPy array to PyTorch tensor
            custom_transform = CustomTransform(self.transforms)
            swir_natural_color, mask = custom_transform(swir_natural_color, mask)

            # Apply the transformations to both image and its mask
            swir_natural_color = ToPILImage()(swir_natural_color)
            mask = ToPILImage()(mask)

            augmented = self.transforms(swir_natural_color, mask)
            swir_natural_color, mask = augmented

            # Convert the transformed PIL images back to PyTorch tensors
            swir_natural_color = ToTensor()(swir_natural_color)
            mask = ToTensor()(mask)

        # Return a tuple of the image and its mask
        return (swir_natural_color, mask)

transforms = [
    Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    ToTensor()
]

# # Create the train and test datasets
# trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
# testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)