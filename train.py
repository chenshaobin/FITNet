from trainUtil.trainingUtil import TrainUtil
from data.readDataFromExcel import getDataFromExcelFile as getOCTFile

if __name__ == '__main__':
    dataSetName = 'Basel'
    kFold = 10
    classNumber = 3
    imgSize = 224
    epoch_num = 100
    baseLR = 1e-3
    lr_scheduler = 'step'
    gpuNumber = "cuda:4"
    lossName = 'crossEntropyLoss'   # or 'focalLoss'
    optimizer = 'SGD'
    # Set the threshold for saving the model
    saveValAcc = 94.0
    saveModelNumber = 50

    encoder = 'FITNet'
    imgDataRoot = 'xxx/ImgData'
    trainExcelFilePath = 'xxx/OCT_Basel_Data.xlsx'
    resultRootPath = 'xxx/Basel/10FoldResult'

    for fold in range(kFold):
        trainExcelSheetName = 'train_fold{}'.format(fold)
        validExcelSheetName = 'valid_fold{}'.format(fold)

        training = TrainUtil(dataSetName=dataSetName, classNumber=classNumber, trainDataRoot=trainExcelSheetName,
                             validDataRoot=validExcelSheetName, encoder=encoder, getDataFunc=getOCTFile,
                             resultRootPath=resultRootPath, baseLR=baseLR, lr_scheduler=lr_scheduler,
                             gpuNumber=gpuNumber, saveValAcc=saveValAcc, saveModelNumber=saveModelNumber,
                             lossName=lossName, optimizer=optimizer, imgSize=imgSize, epoch_num=epoch_num,
                             imgDataRoot=imgDataRoot, trainExcelFilePath=trainExcelFilePath,
                             fold=fold)
        training.running()
