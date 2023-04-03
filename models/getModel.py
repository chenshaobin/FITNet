import torch
import torch.nn as nn
import torchvision
import os

from models.FIT_Net import FITNet

def get_encoder(config, pretrained=False):

    encoder = {
        "FITNet": FITNet(pretrained=pretrained, num_classes=config.classNumber)
    }

    if config.encoder not in encoder.keys():
        raise KeyError(f"{config.encoder} is not a valid model version")
    print('Use model:{}'.format(config.encoder))
    return encoder[config.encoder]
