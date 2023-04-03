from EvaluateUtil.evaluateUtil import EvaluateUtil
from data.readDataFromExcel import getDataFromExcelFile as getOCTFile
from EvaluateMetrics.EvaluateMetricUtils import metric_result_for_eachClass, plotRocCurve_multiClass, plot_confusion_matrix, metric_results_multiClass
from Utils import get_mean_std
from Utils import saveDataAs_npz

if __name__ == '__main__':
    import os
    from os.path import join
    dataSetName = 'Basel'
    classNumber = 3
    labelList = ['Normal', 'DME', 'Other']
    gpuNumber = "cuda:2"
    imgDataRoot = 'xxx/OCT_Basel/ImgData'
    trainExcelFilePath = 'xxx/xxx.xlsx'
    modelRootPath = 'xxx/Basel10FoldResult'
    resultRootPath = 'xxx/metric_Basel'
    if not os.path.isdir(resultRootPath):
        os.makedirs(resultRootPath)
    fileName_npz = join(resultRootPath, 'methodOverallResult.npz')
    result_allFoldsFile = join(resultRootPath, 'result_allFolds.npz')
    result_labelInfo = join(resultRootPath, 'result_labelInfo.npz')
    saveAllFoldsResult = []
    saveOverallResult = []
    saveAllLabelInfo = []
    for method in os.listdir(modelRootPath):
        print('Test:', method)
        saveOverallResultDict = {'method': method}
        modelEncoder = method
        kFoldsModelParam = join(modelRootPath, method, 'model_param')
        result_File = open(join(resultRootPath, '{}_10Fold_result_{}.txt'.format(dataSetName, modelEncoder)), 'a')
        modelList = os.listdir(kFoldsModelParam)
        print('model List:', modelList)
        overallResult = []
        singleResults = []
        totalTrueLabel = []
        totalPreLabel = []
        totalPreLabel_probability = []
        totalScores_AUC = []
        for modelParamFileName in modelList:
            imgSize = 224
            modelPath = join(kFoldsModelParam, modelParamFileName)
            modelFold = modelParamFileName.split(".pth")[0].split('_')[0]
            validExcelSheetName = 'valid_{}'.format(modelFold)
            print('---Test {} ---'.format(validExcelSheetName))

            modelName = modelEncoder
            result_File.write(modelFold + ':' + '\n')
            acc, result, true_labels, pred_labels, scores_AUC = EvaluateUtil(
                testDataRoot=validExcelSheetName, dataSetName=dataSetName, resultRootPath=resultRootPath,
                classNumber=classNumber, modelEncoder=modelEncoder, labelList=labelList,
                getDataFunc=getOCTFile, gpuNumber=gpuNumber, imgSize= imgSize,
                saveCM_and_ROC_Curve_flag=True, useForRocCompare=True,
                imgDataRoot=imgDataRoot, trainExcelFilePath=trainExcelFilePath,
                kFoldFlag=True).main(load_model_weight_path=modelPath, modelName=modelName)

            for result_key, result_value in result.items():
                result_File.writelines([result_key, ':', str(result_value), '\n'])

            singleResult = metric_result_for_eachClass(result['confusion_matrix'])
            singleResults.append(singleResult)
            overallResult.append(result)
            totalTrueLabel.extend(true_labels)
            totalPreLabel.extend(pred_labels)
            totalScores_AUC.extend(scores_AUC)
            print('scores_AUC size:{}, {}'.format(len(scores_AUC), len(scores_AUC[0])))

        print('totalScores_AUC length{}'.format(len(totalScores_AUC)))

        save_path_roc = join(resultRootPath, 'RocResult')
        save_path_CM = join(resultRootPath, 'confusionMatrix')
        if not os.path.isdir(save_path_roc):
            os.makedirs(save_path_roc)
        if not os.path.isdir(save_path_CM):
            os.makedirs(save_path_CM)
        plotRocCurve_multiClass(classNumber, totalTrueLabel, totalScores_AUC, save_path_roc, '{}_{}_roc.png'.format(dataSetName, modelEncoder), labelList)
        result = metric_results_multiClass(classNumber=classNumber, pred_label=totalPreLabel,
                                           gt_label=totalTrueLabel, scores=totalScores_AUC)
        plot_confusion_matrix(result['confusion_matrix'], labelList, join(save_path_CM, '{}_{}'.format(dataSetName, method)))

        totalPreLabel_probability = [totalScores_AUC[index][totalPreLabel[index]] for index in range(len(totalScores_AUC))]
        saveAllLabelInfo.append({'method': method, 'label': totalPreLabel, 'probability': totalPreLabel_probability})

        overallDict = {}
        saveMetricDict = {}
        for key in overallResult[0]:
            if key != 'confusion_matrix':
                overallDict[key] = []
                saveMetricDict[key] = []
        for index, result in enumerate(overallResult):
            for key, value in result.items():
                if key != 'confusion_matrix':
                    overallDict[key].append(value)
        result_File.write('overall result' + ':' + '\n')
        for key, value in overallDict.items():
            mean = get_mean_std(value)['mean']
            std = get_mean_std(value)['std']
            saveMetricDict[key].append(mean)
            saveMetricDict[key].append(std)
            print(f"overall {key}---> mean{mean}, std:{std}")
            result_File.write(key + '--->' + 'mean:' + str(mean) + 'std:' + str(std) + '\n')
        overallDict['method'] = modelEncoder
        print(f"overallDict:{overallDict}")
        saveAllFoldsResult.append(overallDict)
        del overallDict
        saveOverallResultDict['Overall'] = saveMetricDict
        print(f"OverallResult:{saveOverallResultDict}")
        del saveMetricDict
    # for every category
    # """
        result_0 = []
        result_1 = []
        result_2 = []
        for foldIndex, resultList in enumerate(singleResults):
            result_0.append(resultList[0])
            result_1.append(resultList[1])
            result_2.append(resultList[2])
        singleClassResultDict = {'Normal': result_0, 'DME': result_1, 'AMD': result_2}
        print(f"method:{modelEncoder}, Normal result:{singleClassResultDict['Normal']}")

        for label, resultData in singleClassResultDict.items():
            singleResultDict = {}
            saveMetricDict = {}
            for key in singleClassResultDict[label][0]:
                saveMetricDict[key] = []
                singleResultDict[key] = []
            for dictData in resultData:
                for itemKey, value in dictData.items():
                    singleResultDict[itemKey].append(value)
            result_File.write(label + ':' + '\n')
            for metric, value in singleResultDict.items():
                mean = get_mean_std(value)['mean']
                std = get_mean_std(value)['std']
                saveMetricDict[metric].append(mean)
                saveMetricDict[metric].append(std)
                # print(f"label:{label}, {metric}, mean:{mean}, std:{std}")
                result_File.write(metric + ',mean:' + str(mean) + ',std:' + str(std))
            saveOverallResultDict[label] = saveMetricDict
            print(f"{label} result:{saveMetricDict}")
        print(f'saveOverallResultDict:{saveOverallResultDict}')
        saveOverallResult.append(saveOverallResultDict)
        del saveOverallResultDict
        # saveOverallResultDict.clear()
        print(f"saveOverallResultList[-1]:{saveOverallResult[-1]}")
        result_File.close()
        # """

    saveDataAs_npz(fileName=fileName_npz, data=saveOverallResult)
    saveDataAs_npz(fileName=result_allFoldsFile, data=saveAllFoldsResult)
    saveDataAs_npz(fileName=result_labelInfo, data=saveAllLabelInfo)
