# USAGE
# python train.py
# import the necessary packages
import numpy as np
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import glob

class CustomTransform:
     def __init__(self, transforms):
        self.transforms = transforms

     def __call__(self, image, mask):
        image = (image * 255).astype(np.uint8)

        # Convert NumPy arrays to PIL Images
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        # Apply each transform to image and mask
        for transform in self.transforms.transforms:
            image_pil = transform(image_pil)
            mask_pil = transform(mask_pil)

        # Convert PIL Images back to float32
        image = np.array(image_pil) / 255.0
        mask = np.array(mask_pil)

        return image, mask

# load the image and mask filepaths in a sorted manner
imagePaths = glob.glob("/workspaces/U-Net-Training/dataset/train/images/*.tif")
maskPaths = glob.glob("/workspaces/U-Net-Training/dataset/train/mask/*.tif")
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
# Check if the paths are not empty
if not imagePaths or not maskPaths:
    raise ValueError("No image or mask files found.")

# Split the dataset
from sklearn.model_selection import train_test_split

# Split the dataset only if there are samples
if imagePaths and maskPaths:
    imagePaths_train, imagePaths_val, maskPaths_train, maskPaths_val = train_test_split(
        imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42
    )
else:
    raise ValueError("No samples available for train-test split.")

# split = train_test_split(imagePaths, maskPaths,
# 	test_size=config.TEST_SPLIT, random_state=42)
# # unpack the data split
# (trainImages, testImages) = split[:2]
# (trainMasks, testMasks) = split[2:]


# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
with open(config.TEST_PATHS, "w") as f:
    f.write("\n".join(imagePaths_val))

# define transformations
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])
trainDS = SegmentationDataset(imagePaths=imagePaths_train, maskPaths=maskPaths_train, transforms=custom_transforms)
testDS = SegmentationDataset(imagePaths=imagePaths_val, maskPaths=maskPaths_val, transforms=custom_transforms)


# # create the train and test datasets
# trainDS = SegmentationDataset(imagePaths=imagePaths_train, maskPaths=maskPaths_train, transforms=transforms)
# testDS = SegmentationDataset(imagePaths=imagePaths_val, maskPaths=maskPaths_val, transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(config.DEVICE)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
# serialize the model to disk
torch.save(unet, config.MODEL_PATH)
