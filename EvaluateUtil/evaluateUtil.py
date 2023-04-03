from __future__ import print_function
import os
import torch
import numpy as np
from typing import List
import argparse
import torch.utils.data
from torch.autograd import Variable
from models.getModel import get_encoder
import time
import timeit
from tqdm import tqdm
from EvaluateMetrics.EvaluateMetricUtils import plot_confusion_matrix, metric_results, \
    printMetricResults, plotRocCurve, plotRocCurve_multiClass, metric_results_multiClass
import matplotlib.pyplot as plt


class EvaluateUtil:
    def __init__(self,
                 testDataRoot: str, dataSetName: str, resultRootPath: str,
                 classNumber: int, modelEncoder: str, labelList: List,
                 getDataFunc, gpuNumber="cuda:3", dataSetFlag=None,
                 saveCM_and_ROC_Curve_flag=True, useForRocCompare=False,
                 imgSize=224, preWeight=None, imgDataRoot=None,
                 trainExcelFilePath=None, kFoldFlag=None, noise=None, imgNameFlag=None):
        """
            # labelList: Use in confusion matrix and ROC.
            # saveCM_and_ROC_Curve_flag: flag to save confusion matrix and ROC Curve.
            # useForRocCompare: when set True, will return true label and model output scores to compare ROC.
        """
        self.testDataRoot = testDataRoot
        self.imgSize = imgSize
        self.dataSetName = dataSetName
        self.resultRootPath = resultRootPath
        self.modelEncoder = modelEncoder
        self.labelList = labelList
        self.gpuNumber = gpuNumber
        self.classNumber = classNumber
        self.dataSetFlag = dataSetFlag
        self.getData = getDataFunc
        self.saveCM_and_ROC_Curve_flag = saveCM_and_ROC_Curve_flag
        self.useForRocCompare = useForRocCompare
        self.device = torch.device(gpuNumber if torch.cuda.is_available() else "cpu")
        self.preWeight = preWeight
        self.imgDataRoot = imgDataRoot
        self.trainExcelFilePath = trainExcelFilePath
        self.kFoldFlag = kFoldFlag
        self.noise = noise
        self.imgNameFlag = imgNameFlag

    def getParser(self):
        parser = argparse.ArgumentParser(description='OCT Cross-View Testing!')
        parser.add_argument('--testDataRoot', default=self.testDataRoot, type=str, help='test data path.')
        parser.add_argument('--confusionMatrixPath', default='{}/confusionMatrix'.format(self.resultRootPath), type=str,
                            help="save confusionMatrix Path.")
        parser.add_argument('--model_param_MainPath', default='{}/model_param'.format(self.resultRootPath), type=str,
                            help="model param Path.")
        parser.add_argument('--rocCurve_path', default='{}/ROC_Curve'.format(self.resultRootPath), type=str,
                            help="save ROC curve Path.")
        parser.add_argument('--num_worker', default=8, type=int, help='data loader worker number.')
        parser.add_argument('--batchSize', default=64, type=int, help='data loader batch size.')
        parser.add_argument('--encoder', default=self.modelEncoder, type=str, help='Model encoder.')
        parser.add_argument('--dataSetName', default=self.dataSetName, type=str, help='dataSetName.')
        parser.add_argument('--gradCamSavePath', default='{}/gradCamPath'.format(self.resultRootPath), type=str,
                            help="gradCam result Path.")
        args = parser.parse_args()
        args.device = self.device
        args.labelList = self.labelList
        args.modelEncoder = self.modelEncoder
        args.classNumber = self.classNumber
        args.preWeight = self.preWeight
        args.trainExcelFilePath = self.trainExcelFilePath
        args.imgDataRoot = self.imgDataRoot
        if self.saveCM_and_ROC_Curve_flag:
            if not os.path.isdir(args.confusionMatrixPath):
                os.makedirs(args.confusionMatrixPath)
            if not os.path.isdir(args.rocCurve_path):
                os.makedirs(args.rocCurve_path)
        return args

    def main(self, **kwargs):
        config = self.getParser()
        for key, value in kwargs.items():
            setattr(config, key, value)
        start = timeit.default_timer()
        print('Start test model: {} :'.format(config.modelName))

        if self.kFoldFlag is not None:
            if self.dataSetFlag == 'H' or self.dataSetFlag == 'V' \
                    or self.dataSetFlag == 'Horizontal' or self.dataSetFlag == 'Vertical':
                testSet = self.getData(excelFilePath=config.trainExcelFilePath, imgRootPath=config.imgDataRoot,
                                       excelSheetName=config.testDataRoot, dataFlag=self.dataSetFlag, trainFlag=False)
            else:
                if self.noise is not None:
                    testSet = self.getData(excelFilePath=config.trainExcelFilePath, imgRootPath=config.imgDataRoot,
                                           excelSheetName=config.testDataRoot, trainFlag=False, noise=self.noise, imgSize=self.imgSize)
                else:
                    testSet = self.getData(excelFilePath=config.trainExcelFilePath, imgRootPath=config.imgDataRoot,
                                           excelSheetName=config.testDataRoot, trainFlag=False, imgSize=self.imgSize)
        else:
            testSet = self.getData(txtFilePath=config.testDataRoot, dataSetFlag=self.dataSetFlag, imgSize=self.imgSize,
                                   test=True)

        testloader = torch.utils.data.DataLoader(testSet, batch_size=config.batchSize, shuffle=False, num_workers=config.num_worker)

        test_correct = 0
        total = 0
        pred_labels = []
        true_labels = []
        imgNameList = []
        scores = []
        scores_AUC = []

        model = get_encoder(config, pretrained=False)
        model.eval()
        if config.load_model_weight_path:
            print("Loading model param!")
            model.load_state_dict(
                torch.load(config.load_model_weight_path,
                           map_location=lambda storage,
                                               loc: storage.cuda(int(self.gpuNumber.split(':')[-1])))
                # torch.load(config.load_model_weight_path)
            )
        else:
            Exception("No model Param!")
        model.to(config.device)

        for i_val, (img, real_diag, imgName) in tqdm(enumerate(testloader)):
            with torch.no_grad():
                target = real_diag
                img = img.to(config.device)
                target = target.to(config.device)
                img = Variable(img)
                target = Variable(target)
                batch_size = img.size(0)

                score = model(img)
            m = torch.nn.Softmax(dim=1)
            outputs_ = m(score)
            predict_label = score.data.max(1)[1].cpu().numpy()
            imgNameList.extend(imgName)
            scores_AUC.extend(outputs_.data.cpu().numpy().tolist())
            true_label = target.data.cpu().numpy()
            total += batch_size
            test_correct += np.sum(predict_label == true_label)
            pred_labels.extend(predict_label)
            true_labels.extend(true_label)
            scores.extend(score.data.cpu().numpy().tolist())

        test_acc = float(test_correct) / total
        acc = test_acc * 100
        if config.classNumber == 2:
            result = metric_results(pred_labels, true_labels, scores_AUC)
        else:
            result = metric_results_multiClass(classNumber=config.classNumber, pred_label=pred_labels,
                                               gt_label=true_labels, scores=scores_AUC)
        printMetricResults(result)
        if self.saveCM_and_ROC_Curve_flag:
            plt.clf()
            print('Save confusion matrix and ROC Curve!')
            confusion_matrixPath = os.path.join(
                config.confusionMatrixPath, '%s_%s_acc_%f.jpg' % (config.dataSetName, config.modelName, acc))
            plot_confusion_matrix(result['confusion_matrix'], classes=config.labelList,
                                  normalize=False, title='', savePath=confusion_matrixPath)
            picName = '{}_{}.png'.format(config.modelName, acc)
            plt.clf()
            if config.classNumber == 2:
                plotRocCurve(true_labels, scores_AUC, config.rocCurve_path, picName, picLabel=config.dataSetName)
            else:
                plotRocCurve_multiClass(classNumber=config.classNumber, true_label=true_labels,
                                        scores=scores_AUC, save_path=config.rocCurve_path,
                                        picName=picName, picLabel=config.labelList)
        else:
            print('Don\'t Save confusion matrix and ROC Curve!')
        end = timeit.default_timer()
        print('Finishing testing {}, Get Acc:{:.3f}%, total have {} samples, run {} seconds '
              .format(config.modelName, acc, total, (end - start)))
        if self.useForRocCompare and self.kFoldFlag is not None and self.imgNameFlag is None:
            print(f"scores_AUC size:{len(scores_AUC)}, shape:{len(scores_AUC[0])}")
            return acc, result, true_labels, pred_labels, scores_AUC
        elif self.useForRocCompare and self.kFoldFlag is not None and self.imgNameFlag:
            print(f"scores_AUC size:{len(scores_AUC)}, shape:{len(scores_AUC[0])}, imgNameShape:{len(imgNameList)}")
            return acc, result, true_labels, pred_labels, scores_AUC, imgNameList
        else:
            return acc, result

    def batch_save_result(self):
        config = self.getParser()
        result_File = open(
            os.path.join(config.confusionMatrixPath,
                         '{}_result_{}.txt'.format(config.dataSetName, config.modelEncoder)), 'w')
        model_param_list = os.listdir(config.model_param_MainPath)
        result_Lists = []
        for index, item in enumerate(model_param_list):
            print('Processing model:{}/{}'.format(index + 1, len(model_param_list)))
            result_List = {'Name': item.split('.pth')[0]}
            top1_avg, result = self.main(
                load_model_weight_path=os.path.join(config.model_param_MainPath, item), modelName=result_List['Name'])
            result_List['acc'] = top1_avg
            result_List['result'] = result
            result_Lists.append(result_List)
            del result_List
        result_Lists.sort(key=lambda k: k.get('acc'), reverse=True)
        for result in result_Lists:
            result_File.write('\n' + result['Name'] + ':' + '\n')
            for result_key, result_value in result['result'].items():
                result_File.writelines([result_key, ':', str(result_value), '\n'])

        result_File.close()

    def running(self):
        import fire
        fire.Fire()
        self.batch_save_result()
