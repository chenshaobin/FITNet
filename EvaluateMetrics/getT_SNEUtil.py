from models.getModel import get_encoder
from EvaluateMetrics.EvaluateMetricUtils import get_feature, getT_SNE
from typing import List
import argparse
import torch
from tqdm import tqdm
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from os.path import join
import os

class getTSNEUtil:
    def __init__(self, testDataRoot, resultRootPath, classNumber: int,
                 modelEncoder: str, labelList: List, getDataFunc,
                 imgSize: int, modelParamPath: str, dataSetFlag=None,
                 target_layers=['avgpool'], gpuNumber="cuda:3", batchSize=1,
                 kFold=1, imgDataRoot=None, readExcelFileFlag=False,
                 excelFilePath=None, savePath=None, otherFlag=None):
        self.testDataRoot = testDataRoot
        self.resultRootPath = resultRootPath
        self.classNumber = classNumber
        self.modelEncoder = modelEncoder
        self.labelList = labelList
        self.gpuNumber = gpuNumber
        self.getDataFunc = getDataFunc
        self.imgSize = imgSize
        self.dataSetFlag = dataSetFlag
        self.modelParamPath = modelParamPath
        self.batchSize = batchSize
        self.target_layers = target_layers
        self.device = torch.device(gpuNumber if torch.cuda.is_available() else "cpu")
        self.kFold = kFold
        self.imgDataRoot = imgDataRoot
        self.readExcelFileFlag = readExcelFileFlag
        self.excelFilePath = excelFilePath
        self.savePath = savePath
        self.otherFlag = otherFlag

    def getParser(self):
        parser = argparse.ArgumentParser(description='Generate t-SNE!')
        parser.add_argument('--batchSize', default=self.batchSize, type=int)
        parser.add_argument('--num_worker', default=8, type=int, help='data loader worker number.')
        parser.add_argument('--testDataRoot', default=self.testDataRoot, type=str)
        parser.add_argument('--resultRootPath', default=self.resultRootPath, type=str)
        parser.add_argument('--classNumber', default=self.classNumber, type=int)
        parser.add_argument('--encoder', default=self.modelEncoder, type=str)
        parser.add_argument('--imgSize', default=self.imgSize, type=int)
        parser.add_argument('--modelParamPath', default=self.modelParamPath, type=str, help="model param Path.")
        args = parser.parse_args()
        args.device = self.device
        args.tSNE_fileSavePath = join(self.resultRootPath, 'tSNE_filePath')
        args.tSNE_figSavePath = join(self.resultRootPath, 'tSNE_figPath')
        args.figName = join(args.tSNE_figSavePath, args.encoder + '_tSEN.png')
        args.save_name = join(args.tSNE_fileSavePath, args.encoder + '_tSEN.npz')
        args.kFold = self.kFold
        args.imgDataRoot = self.imgDataRoot
        args.readExcelFileFlag = self.readExcelFileFlag
        args.excelFilePath = self.excelFilePath
        args.preWeight = None

        return args

    def get_tSNEFile(self):
        config = self.getParser()
        model = get_encoder(config, pretrained=False)
        if config.kFold == 1:
            testData = self.getDataFunc(txtFilePath=config.testDataRoot, dataSetFlag=self.dataSetFlag,
                                        imgSize=self.imgSize, test=True)
            testLoader = torch.utils.data.DataLoader(testData, batch_size=config.batchSize,
                                                     shuffle=False, num_workers=config.num_worker)
            model.eval()
            if config.modelParamPath:
                print("Loading model param!")
                model.load_state_dict(
                    # torch.load(config.modelParamPath)
                    torch.load(config.modelParamPath,
                               map_location=lambda storage, loc: storage.cuda(int(self.gpuNumber.split(':')[-1])))
                )
            else:
                Exception("No model Param!")
            model.to(config.device)
            predLabels = []
            features = []
            for i_val, (img, real_label, _) in tqdm(enumerate(testLoader)):
                with torch.no_grad():
                    target = real_label
                    img = img.to(config.device)
                    target = target.to(config.device)
                    batch_size = img.size(0)
                    score = model(img)
                m = torch.nn.Softmax(dim=1)
                predict_label = score.data.max(1)[1].cpu().numpy()
                predLabels.extend(predict_label)
                feature = get_feature(model, img, target_layers=self.target_layers)
                feature_avg = np.average(feature, axis=0)
                print('feature shape:{}, avg feature shape:{}'.format(feature.shape, feature_avg.shape))
                features.append(feature_avg)

            feature = np.array(features)
            featureLabel = np.array(predLabels)
            save_name = join(config.tSNE_fileSavePath, config.encoder + '_tSEN.npz')
            np.savez(save_name, data=feature, label=featureLabel)
        else:
            predLabels = []
            features = []
            for modelParamFoldName in config.testDataRoot:
                modelFold = modelParamFoldName.split(".pth")[0].split('_')[0]
                validExcelSheetName = 'valid_{}'.format(modelFold)
                print(f"Test {config.encoder}, {validExcelSheetName}")
                testData = self.getDataFunc(excelFilePath=config.excelFilePath, imgRootPath=config.imgDataRoot,
                                            excelSheetName=validExcelSheetName, trainFlag=False)
                testLoader = torch.utils.data.DataLoader(testData, batch_size=config.batchSize, shuffle=False,
                                                         num_workers=config.num_worker)
                model.eval()
                modelPath = join(config.modelParamPath, modelParamFoldName)
                print("Loading model param!")
                model.load_state_dict(
                    # torch.load(config.modelParamPath)
                    torch.load(modelPath,
                               map_location=lambda storage, loc: storage.cuda(int(self.gpuNumber.split(':')[-1])))
                )
                model.to(config.device)
                if self.otherFlag is not None:
                    for i_val, dataDict in tqdm(enumerate(testLoader)):
                        with torch.no_grad():
                            img = dataDict["A"]
                            target = dataDict["label"]
                            img = img.to(config.device)
                            target = target.to(config.device)
                            batch_size = img.size(0)
                            score, _, _, _ = model(img)
                        m = torch.nn.Softmax(dim=1)
                        predict_label = score.data.max(1)[1].cpu().numpy()
                        predLabels.extend(predict_label)
                        feature = get_feature(model, img, target_layers=self.target_layers, modelName=config.encoder)
                        feature_avg = np.average(feature, axis=0)
                        print('feature shape:{}, avg feature shape:{}'.format(feature.shape, feature_avg.shape))
                        features.append(feature_avg)
                else:
                    for i_val, (img, real_label, _) in tqdm(enumerate(testLoader)):
                        with torch.no_grad():
                            target = real_label
                            img = img.to(config.device)
                            target = target.to(config.device)
                            batch_size = img.size(0)
                            score = model(img)
                        m = torch.nn.Softmax(dim=1)
                        predict_label = score.data.max(1)[1].cpu().numpy()
                        predLabels.extend(predict_label)
                        feature = get_feature(model, img, target_layers=self.target_layers, modelName=config.encoder)
                        feature_avg = np.average(feature, axis=0)
                        print('feature shape:{}, avg feature shape:{}'.format(feature.shape, feature_avg.shape))
                        features.append(feature_avg)

            feature = np.array(features)
            featureLabel = np.array(predLabels)
            savePath = join(self.savePath, 'tSNE_filePath')
            if not os.path.isdir(savePath):
                os.makedirs(savePath)
            save_name = join(savePath, config.encoder + '_tSEN.npz')
            np.savez(save_name, data=feature, label=featureLabel)

    def create_tSNE_fig(self):
        config = self.getParser()
        if config.kFold != 1:
            save_name = join(self.savePath, 'tSNE_filePath', config.encoder + '_tSEN.npz')
            figSavePath = join(self.savePath, 'tSNE_figPath')
            if not os.path.isdir(figSavePath):
                os.makedirs(figSavePath)
            figName = join(figSavePath, config.encoder + '_tSEN.png')
            getT_SNE_obj = getT_SNE(save_name, config.classNumber, self.labelList, figName)
        else:
            getT_SNE_obj = getT_SNE(config.save_name, config.classNumber, self.labelList, config.figName)
        getT_SNE_obj.main()
