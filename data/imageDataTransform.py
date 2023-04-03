from torchvision import transforms as T
import torch
import random
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image

"""
# Random cropping of images during training:
    #1: First perform compression to keep the aspect ratio of the image unchanged, 
        and then perform random cropping:Resize_and_RandomCrop.
    #2: Do random cropping directly on the original image
"""

def pad_if_smaller(img, size, fill=0):
    pass


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob: float):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob: float):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
        return image


class Resize(object):
    """
        # If imgSize is an int, the smaller of the image's height and width will match this number while maintaining the image's aspect ratio
        # Suppose the original input image size is (3, 500, 332), imgSize=224, then the output image size is (3, 337, 224)
    """
    def __init__(self, imgSize):
        self.imgSize = imgSize

    def __call__(self, image):
        image = TF.resize(image, self.imgSize)
        return image


class Resize_and_RandomCrop(object):
    """
    # First, the image is compressed to maintain the horizontal and vertical ratio,
    # and then randomly clipped to the specified size.
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        image = Resize(self.size)(image)
        # Gets the parameters for random clipping
        # Pytorch Official statement:https://pytorch.org/vision/0.12/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        return image


class RandomCrop(object):
    """
    # Do a random crop directly on the original image
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        image = Resize(self.size)(image)
        # Gets the parameters for random clipping
        # Pytorch Official statement :https://pytorch.org/vision/0.12/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        return image


class AddSaltPepperNoise(object):

    def __init__(self, density=0.):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) < 3 or (len(img.shape) == 3 and img.shape[2] == 1):
            img_temp = np.zeros((img.shape[0], img.shape[1], 3))
            img_temp[:, :, 0] = img
            img_temp[:, :, 1] = img
            img_temp[:, :, 2] = img
            img = img_temp

        # print(f"img shape:{img.shape}")
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # Generates a mask for a channel
        mask = np.repeat(mask, c, axis=2)                                               # Copy in the dimensions of the channel
        img[mask == 0] = 0
        img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class RandomAddSaltPepperNoise(object):

    def __init__(self, flip_prob: float):
        self.flip_prob = flip_prob
        self.densityList = [0.01, 0.03, 0.05, 0.1]

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) < 3 or (len(img.shape) == 3 and img.shape[2] == 1):
            img_temp = np.zeros((img.shape[0], img.shape[1], 3))
            img_temp[:, :, 0] = img
            img_temp[:, :, 1] = img
            img_temp[:, :, 2] = img
            img = img_temp

        h, w, c = img.shape
        if random.random() < self.flip_prob:
            density = self.densityList[random.randint(0, len(self.densityList) - 1)]
            Nd = density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # Generates a mask for a channel
            mask = np.repeat(mask, c, axis=2)                                               # Copy in the dimensions of the channel
            img[mask == 0] = 0
            img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class ToTensor(object):
    def __call__(self, image):
        image = TF.to_tensor(image)
        return image
