"""
    # Divide the data into k-folds for cross-validation
    # Write the picture name and label of the training and verification data for each fold
    # into different subsheets of the same excel file
    # This example assumes that there are 3 categories for your dataset!
"""
import re
from os.path import join
import os
import pandas as pd
import shutil
from typing import List
from collections import Counter
import random

class get_k_fold_Data:
    def __init__(self, excelFileSavePath: str, imgRootPath: str, kFold: int, num_class=4):
        """
        :param excelFileSavePath: The path of the excel file to be stored
        :param imgRootPath: The image storage path, the images under this path have been classified according to folders,
               and the first letter of the folder name indicates the category number
        :param kFold: The number of folds to divide the data
        :param num_class: Number of dataset categories
        """
        self.imgRootPath = imgRootPath
        self.excelFileSavePath = excelFileSavePath
        self.kFold = kFold
        self.num_class = num_class

    def get_k_fold_index(self, data: List, everyFoldSize: int, kthFold: int) -> List:
        """
        :param everyFoldSize: Number of images per fold
        :param kthFold: [0 , self.kFold - 1]
        :return: Return the corresponding image name
        # if (dataLength % kthFold) != 0, the redundant data is stored in the last fold data
        """
        assert kthFold <= self.kFold - 1, "The fold number :{} is out of range!".format(kthFold)
        dataLength = len(data)
        train = []
        valid = []
        for j in range(self.kFold):
            idx = slice(j * everyFoldSize, (j + 1) * everyFoldSize)
            data_part = data[idx]
            if j == kthFold:
                # The fold of data that belongs to the validation set
                if kthFold == self.kFold - 1:
                    # the last fold data, if(dataLength % kthFold) != 0, the redundant data is stored in the last fold data
                    index = slice(j * everyFoldSize, dataLength)
                    valid = data[index]
                else:
                    valid = data_part
            elif len(train) == 0:
                train = data_part
            elif j == (self.kFold - 1):
                # the last fold data, if(dataLength % kthFold) != 0, the redundant data is stored in the last fold data
                index = slice(j * everyFoldSize, dataLength)
                data_part = data[index]
                train.extend(data_part)
            else:
                train.extend(data_part)

        return [train, valid]

    def createExcelFile(self, dataInfo: List, sheet_name: str):
        excelData = {'img': [item["img"] for item in dataInfo],
                     'label': [item["label"] for item in dataInfo]}
        df = pd.DataFrame(data=excelData)
        with pd.ExcelWriter(self.excelFileSavePath, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name,  index=False)
        writer.save()
        writer.close()

    def getKFoldData(self):
        # Statistical image file name and corresponding label, format:{'img':img1.tif, 'label': 0}
        imgDataInfo = []
        for category in os.listdir(self.imgRootPath):
            categoryPath = join(self.imgRootPath, category)
            for img in os.listdir(categoryPath):
                imgInfo = {'img':img, 'label': category[0]}
                imgDataInfo.append(imgInfo)
        # Write data information into excel file
        df = pd.DataFrame(imgDataInfo)
        df.to_excel(excelFileSavePath, index=False, sheet_name='originalFile')
        categoryDict = {}
        for classIndex in range(self.num_class):
            classLabel = str(classIndex)
            categoryDict[classLabel] = []
        for dataInfo in imgDataInfo:
            categoryDict[dataInfo['label']].append(dataInfo)

        # print('Category statistics:0:{}, 1:{}, 2:{}'.format(len(category_0), len(category_1), len(category_2)))
        for label, value in categoryDict.items():
            print(f"{label}:{len(value)}")
        categoryFoldSizeDict = {}
        for label in categoryDict.keys():
            categoryFoldSizeDict[label] = int(len(categoryDict[label]) / self.kFold)
            print(f"{label}:{categoryFoldSizeDict[label]}")
        oneFoldSize = 0
        for label in categoryFoldSizeDict.keys():
            oneFoldSize += categoryFoldSizeDict[label]
        print(f"one fold size:", oneFoldSize)

        for kthFold in range(self.kFold):
            fold_train = []
            fold_valid = []
            for label, value in categoryDict.items():
                train, valid = self.get_k_fold_index(categoryDict[label], categoryFoldSizeDict[label], kthFold)
                fold_train.extend(train)
                fold_valid.extend(valid)
                assert len(value) == len(train) + len(valid), \
                    f'Category:{label}The original quantity:{len(value)}, after k folds:{len(train) + len(valid)}'

            print('fold:{} train size:{}'.format(kthFold, len(fold_train)))
            print('fold:{} valid size:{}'.format(kthFold, len(fold_valid)))
            assert len(imgDataInfo) == (len(fold_train) + len(fold_valid)),\
                'The amount of data is inconsistent, please check! --> The original quantity:{}, after k folds:{}'.format(len(imgDataInfo), len(fold_train)+len(fold_valid))

            train_sheetName = 'train_' + 'fold' + str(kthFold)
            print('train sheetName:', train_sheetName)
            self.createExcelFile(fold_train, train_sheetName)

            valid_sheetName = 'valid_' + 'fold' + str(kthFold)
            print('valid sheetName:', valid_sheetName)
            self.createExcelFile(fold_valid, valid_sheetName)


if __name__ == '__main__':
    # Get k-fold data
    excelFileSavePath = r'xxx\xxx.xlsx'
    imgRootPath = r'xxx\imgPath'
    kFold = 10
    mainObj = get_k_fold_Data(excelFileSavePath=excelFileSavePath, imgRootPath=imgRootPath, kFold=kFold, num_class=4)
    mainObj.getKFoldData()
