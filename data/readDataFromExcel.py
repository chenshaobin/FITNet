import numpy as np
import os
from PIL import Image
# import imageio
from torchvision import transforms as T
from data.imageDataTransform import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize_and_RandomCrop, \
    RandomCrop, ToTensor, AddSaltPepperNoise, RandomAddSaltPepperNoise
from torch.utils import data
from typing import List, Dict, Any
import pandas as pd
from os.path import join
import torch
# For your data set, you can use Utils.py->Get_mean_std_for_dataset to calculate
octFile_MEAN = np.array([0.193, 0.193, 0.193], dtype=np.float32)
octFile_STD = np.array([0.201,  0.201, 0.201], dtype=np.float32)

def getOctTrainTransforms(imgSize=224):
    octTrainImgTransforms = Compose([
        Resize_and_RandomCrop(size=imgSize),
        # RandomCrop(size=224),
        RandomHorizontalFlip(0.5),
        # RandomVerticalFlip(0.5),
        ToTensor(),
        T.Normalize(mean=octFile_MEAN.tolist(), std=octFile_STD.tolist())
    ])
    return octTrainImgTransforms

def getOctTestTransforms(imgSize=224):
    octTestImgTransform = Compose([
        T.Resize(size=(224, 224)),
        ToTensor(),
        T.Normalize(mean=octFile_MEAN.tolist(), std=octFile_STD.tolist())
    ])
    return octTestImgTransform

octTrainImgTransforms_withNoise = Compose([
    RandomAddSaltPepperNoise(0.4),
    Resize_and_RandomCrop(size=224),
    # RandomCrop(size=224),
    RandomHorizontalFlip(0.5),
    # RandomVerticalFlip(0.5),
    ToTensor(),
    T.Normalize(mean=octFile_MEAN.tolist(), std=octFile_STD.tolist())
])

def octTestImgTransform_withNoise(density):
    octTestImgTransform = Compose([
        AddSaltPepperNoise(density),
        T.Resize(size=(224, 224)),
        ToTensor(),
        T.Normalize(mean=octFile_MEAN.tolist(), std=octFile_STD.tolist())
    ])
    return octTestImgTransform

def floatToTensor(floatValue):
    floatValue_tensor = torch.as_tensor(floatValue, dtype=torch.float32)
    return floatValue_tensor


class getDataFromExcelFile(data.Dataset):
    def __init__(self, excelFilePath: str, imgRootPath: str, excelSheetName: str, trainFlag=True, noise=None, imgSize=224):
        super(getDataFromExcelFile, self).__init__()
        self.excelFilePath = excelFilePath
        self.imgRootPath = imgRootPath
        self.excelSheetName = excelSheetName
        self.excelData = self.readCheckedExcel()
        self.trainFlag = trainFlag
        self.noise = noise
        self.imgSize = imgSize

    def readCheckedExcel(self) -> List:
        """
            # Read the organized annotation information from excel table
        """
        final_list = []
        excelData = pd.read_excel(self.excelFilePath, sheet_name=self.excelSheetName)
        excelData = excelData.dropna(
            subset=['img', 'label'])
        imgName = excelData.loc[:, 'img'].values
        imgLabel = excelData.loc[:, 'label'].values

        dataLen = len(imgName)
        print('Total amount of data:', dataLen)
        item_dict = {}
        for index in range(dataLen):
            # Get image and label information
            item_dict["img"] = imgName[index]
            item_dict["label"] = imgLabel[index]
            final_list.append(item_dict)
            item_dict = {}
        print('The amount of data organized in Excel table:', len(final_list))
        return final_list

    def __len__(self):
        return len(self.excelData)

    def __getitem__(self, index) -> List:
        imgPath = join(self.imgRootPath, self.excelData[index]["img"])
        imgLable = self.excelData[index]['label']
        imgFile = Image.open(imgPath).convert('RGB')

        if self.trainFlag:
            if self.noise is None:
                imgFile = getOctTrainTransforms(self.imgSize)(imgFile)
            else:
                imgFile = octTrainImgTransforms_withNoise(imgFile)
        else:
            if self.noise is not None:
                imgFile = octTestImgTransform_withNoise(self.noise)(imgFile)
            else:
                imgFile = getOctTestTransforms(self.imgSize)(imgFile)

        return [imgFile, imgLable, os.path.basename(self.excelData[index]["img"]).split(".")[0]]

# Just for PM OCT Dual-view lesion classification
class getDataFromExcelFile_HV(data.Dataset):
    def __init__(self, excelFilePath: str, imgRootPath: str, excelSheetName: str, dataFlag: str, trainFlag=True, imgSize=224):
        super(getDataFromExcelFile_HV, self).__init__()
        self.excelFilePath = excelFilePath
        self.imgRootPath = imgRootPath
        self.excelSheetName = excelSheetName
        self.eyePairData, self.imgData = self.readCheckedExcel()
        self.dataFlag = dataFlag
        self.trainFlag = trainFlag
        self.imgSize = imgSize

    def readCheckedExcel(self) -> tuple:
        """
            # Read the organized annotation information from excel table
        """
        eyePairData = []
        imgData = []
        excelData = pd.read_excel(self.excelFilePath, sheet_name=self.excelSheetName)
        excelData = excelData.dropna(subset=['Himg', 'Vimg', 'label'])
        hImgName = excelData.loc[:, 'Himg'].values
        vImgName = excelData.loc[:, 'Vimg'].values
        imgLabel = excelData.loc[:, 'label'].values
        dataLen = len(imgLabel)
        print('Total amount of data:', dataLen)
        for index in range(dataLen):
            eyePairData.append({'Himg': hImgName[index], 'Vimg': vImgName[index], 'label': imgLabel[index]})
            imgData.append({'img': hImgName[index], 'label': imgLabel[index]})
            imgData.append({'img': vImgName[index], 'label': imgLabel[index]})
        print(f"dataSet eye number:{len(eyePairData)}, total img number:{len(imgData)}")
        return eyePairData, imgData

    def __len__(self):
        if self.dataFlag == 'HV':
            return len(self.imgData)
        else:
            return len(self.eyePairData)

    def __getitem__(self, index):
        if self.dataFlag == 'HVFusion' or self.dataFlag == 'H' or self.dataFlag == 'V'\
                or self.dataFlag == 'Horizontal' or self.dataFlag == 'Vertical':
            hImg = Image.open(join(self.imgRootPath, self.eyePairData[index]['Himg'])).convert('RGB')
            vImg = Image.open(join(self.imgRootPath, self.eyePairData[index]['Vimg'])).convert('RGB')
            label = self.eyePairData[index]['label']
            if self.trainFlag:
                hImg = getOctTrainTransforms(self.imgSize)(hImg)
                vImg = getOctTrainTransforms(self.imgSize)(vImg)
            else:
                hImg = getOctTestTransforms(self.imgSize)(hImg)
                vImg = getOctTestTransforms(self.imgSize)(vImg)
            if self.dataFlag == 'HVFusion':
                return hImg, vImg, label, self.eyePairData[index]['Himg'], self.eyePairData[index]['Vimg']
            elif self.dataFlag == 'H' or self.dataFlag == 'Horizontal':
                return hImg, label, self.eyePairData[index]['Himg']
            else:
                return vImg, label, self.eyePairData[index]['Vimg']
        elif self.dataFlag == 'HV':
            imgPath = join(self.imgRootPath, self.imgData[index]['img'])
            imgLabel = self.imgData[index]['label']
            imgFile = Image.open(imgPath).convert('RGB')
            if self.trainFlag:
                imgFile = getOctTrainTransforms(self.imgSize)(imgFile)
            else:
                imgFile = getOctTestTransforms(self.imgSize)(imgFile)
            return imgFile, imgLabel, self.imgData[index]['img']
        else:
            Exception(f'error dataFlag:{self.dataFlag}, choose in [H, V, HV, HVFusion]')


if __name__ == '__main__':
    excelFilePath = '/xxx/xxxx.xlsx'
    imgRootPath = 'xxx/xxx'
    excelSheetName = 'train_fold0'
    mainObj = getDataFromExcelFile(excelFilePath=excelFilePath, imgRootPath=imgRootPath, excelSheetName=excelSheetName)
    img, label = mainObj[0]
    print('img shape:{}, label:{}'.format(img.shape, label))


