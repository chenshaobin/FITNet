import os
import torch.nn.functional as F
import logging
from tensorboardX import SummaryWriter          # visualization module
from torchnet import meter
from tqdm import tqdm  # progress bar tool
from sklearn.metrics import confusion_matrix
import timeit
import torch
import torch.utils.data as torchdata
import argparse
from models.getModel import get_encoder
from Utils import create_lr_scheduler, createLossFunc, create_optimizer
import time


class TrainUtil:
    def __init__(self, dataSetName: str, classNumber: int, trainDataRoot: str, validDataRoot: str,
                 encoder: str, getDataFunc, resultRootPath='/xxx/ExperimenResult',
                 baseLR=1e-3, lr_scheduler='step', gpuNumber="cuda:3", saveValAcc=95.0,
                 saveModelNumber=100, lossName='crossEntropyLoss', optimizer='Adam', dataSetFlag=None,
                 imgSize=224, epoch_num=200, preWeight=None, imgDataRoot=None, trainExcelFilePath=None,
                 teacher=None, fold=None, noise=None, batchSize=64, focalLossGamma=2):
        self.dataSetName = dataSetName
        self.classNumber = classNumber
        self.epoch_num = epoch_num
        self.getData = getDataFunc
        self.trainDataRoot = trainDataRoot
        self.validDataRoot = validDataRoot
        self.baseLR = baseLR
        self.lr_scheduler = lr_scheduler
        self.modelEncoder = encoder
        self.gpuNumber = gpuNumber
        self.imgSize = imgSize
        self.device = torch.device(gpuNumber if torch.cuda.is_available() else "cpu")
        self.timeInfo = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        self.resultRootPath = resultRootPath
        self.saveValAcc = saveValAcc
        self.lossName = lossName
        self.saveModelNumber = saveModelNumber
        self.optimizer = optimizer
        self.dataSetFlag = dataSetFlag
        self.preWeight = preWeight
        self.imgDataRoot = imgDataRoot
        self.trainExcelFilePath = trainExcelFilePath
        self.teacher = teacher
        self.fold = fold
        self.batchSize = batchSize
        self.noise = noise
        self.focalLossGamma = focalLossGamma

    def getParser(self):
        parser = argparse.ArgumentParser(description='Model Training!')
        parser.add_argument('--trainDataRoot', default=self.trainDataRoot, type=str, help='train data path.')
        parser.add_argument('--validDataRoot', default=self.validDataRoot, type=str, help='valid data path.')
        parser.add_argument('--evaluationFrq', default=1, type=int, help='every evaluationFrq epoch to do a evaluation')
        parser.add_argument('--warmup_epochs', default=8, type=int, help='Every warmup_epochs epoch, the learning rate is multiplied by gamma to perform the attenuation operation')
        parser.add_argument('--num_worker', default=8, type=int, help='data loader worker number.')
        parser.add_argument('--encoder', default=self.modelEncoder, type=str, help='Model encoder.')
        parser.add_argument('--dataSetName', default=self.dataSetName, type=str, help='dataSetName.')
        parser.add_argument('--lrf', type=float, default=0.01)

        args = parser.parse_args()
        args.baseLR = self.baseLR
        args.batchSize = self.batchSize
        args.lr_scheduler = self.lr_scheduler
        args.imgSize = self.imgSize
        args.device = self.device
        args.saveValAcc = self.saveValAcc
        args.saveModelNumber = self.saveModelNumber
        args.lossName = self.lossName
        args.optimizer = self.optimizer
        args.classNumber = self.classNumber
        args.preWeight = self.preWeight
        args.imgDataRoot = self.imgDataRoot
        args.trainExcelFilePath = self.trainExcelFilePath
        args.teacher = self.teacher
        args.focalLossGamma = self.focalLossGamma
        if self.fold is not None:
            args.rootPath = '{}/{}/fold{}_{}_{}_lr_{}_{}_{}_{}'.format(
                self.resultRootPath, args.encoder, self.fold, self.dataSetName, self.timeInfo, args.baseLR, args.lr_scheduler,
                self.lossName, self.optimizer)

            args.bestModelParamDir = '{}/{}/model_param'.format(self.resultRootPath, args.encoder)
            if not os.path.isdir(args.bestModelParamDir):
                os.makedirs(args.bestModelParamDir)
        else:
            args.rootPath = '{}/{}/{}_{}_lr_{}_{}_{}_{}'.format(
                self.resultRootPath, args.encoder, self.dataSetName, self.timeInfo, args.baseLR, args.lr_scheduler, self.lossName, self.optimizer)

        args.logfile_path = '{}/LogFiles'.format(args.rootPath)
        args.model_params_dir = '{}/model_param'.format(args.rootPath)
        args.tensorboardX_params_dir = '{}/tensorboard'.format(args.rootPath)

        if not os.path.isdir(args.logfile_path):
            os.makedirs(args.logfile_path)
        if not os.path.isdir(args.model_params_dir):
            os.makedirs(args.model_params_dir)
        if not os.path.isdir(args.tensorboardX_params_dir):
            os.makedirs(args.tensorboardX_params_dir)

        return args

    @staticmethod
    def trainlog(logfilepath, head='%(message)s'):
        logger = logging.getLogger('mylogger')
        logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(head)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def evaluationMode(self, model, dataloader, criterion, config):
        model.eval()  # Set model to evaluate mode
        val_loss_meter = meter.AverageValueMeter()
        pred = []
        true = []

        val_loss = 0
        val_corrects = 0
        val_total = 0
        for batch_cnt, (img, label, _) in tqdm(enumerate(dataloader)):
            val_loss_meter.reset()
            Imgs, labels = img, label
            Imgs = Imgs.to(config.device)
            labels = labels.to(config.device)
            batchsize = Imgs.size(0)
            # forward
            """
            if config.encoder == 'InceptionV3':
                outputs, _ = model(Imgs)
            else:
                outputs = model(Imgs)
            """
            outputs = model(Imgs)

            loss_1 = criterion(outputs, labels)
            loss = loss_1

            val_loss_meter.add(loss.cpu().data)
            label_pred = outputs.data.max(1)[1].cpu().numpy()
            label_true = labels.cpu().data.numpy()
            pred.extend(label_pred)
            true.extend(label_true)

            _, preds = torch.max(outputs, 1)
            correct = torch.sum(preds.data == labels.data)
            val_corrects += correct
            # statistics
            val_loss += loss.item()
            val_total += batchsize

        confusionMatrix = confusion_matrix(true, pred)
        accuracy = 100. * float(val_corrects) / val_total
        val_loss = val_loss / val_total

        return val_loss, accuracy, confusionMatrix

    def train(self, model, epoch_num, start_epoch, optimizer,
              LossFuc, exp_lr_scheduler, train_loader, valid_loader,
              model_params_dir, tensorboardX_dir, config, teacherModel=None
              ):
        writer = SummaryWriter(tensorboardX_dir)
        loss_meter = meter.AverageValueMeter()  # Record the mean and variance of the loss function

        step = -1
        start = timeit.default_timer()  # start timer
        saveModelNum = 0
        bestAcc = 0.0
        for epoch in range(start_epoch, epoch_num + 1):
            loss_meter.reset()
            print('Epoch:', epoch, '| lr: %s' % exp_lr_scheduler.get_last_lr())
            model.train(True)  # Set model to training mode
            total = 0.0
            train_correct = 0.0
            train_loss = 0.0
            for batch_cnt, (img, label, _) in tqdm(enumerate(train_loader)):  # Add progress bar tool
                step += 1
                model.train(True)
                Imgs, labels = img, label
                Imgs = Imgs.to(config.device)
                labels = labels.to(config.device)
                batch_size = Imgs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()
                if teacherModel is not None:
                    with torch.no_grad():
                        teacher_logits = teacherModel(Imgs)
                if config.encoder == 'InceptionV3':
                    outputs, _ = model(Imgs)
                else:
                    if teacherModel is not None:
                        outputs, distill_logits = model(Imgs)
                    else:
                        outputs = model(Imgs)

                loss_1 = LossFuc(outputs, labels)
                if teacherModel is not None:
                    distill_loss = F.kl_div(
                        F.log_softmax(distill_logits / 3, dim=-1),
                        F.softmax(teacher_logits / 3, dim=-1).detach(), reduction='batchmean')
                    distill_loss *= 3 ** 2
                    loss = loss_1 * 0.5 + distill_loss * 0.5
                else:
                    loss = loss_1

                loss.backward()
                optimizer.step()
                loss_meter.add(loss.cpu().data)

                if step % 100 == 0:
                    print('batch:', step, '| train loss(Average loss per 100 batches): %.6f' % loss_meter.value()[0])
                    writer.add_scalar("train/loss", loss_meter.value()[0], step)

                _, preds = torch.max(outputs, 1)
                correct = torch.sum(preds.data == labels.data)
                batch_acc = float(correct) / batch_size
                total += batch_size
                train_correct += correct
                train_loss += loss.item() * batch_size

            # Record the loss function and accuracy after training an epoch
            train_acc = float(train_correct) / total
            average_loss = train_loss / total
            logging.info('[%d] | train-epoch-loss: %.3f | acc@1: %.3f'
                         % (epoch, average_loss, train_acc))
            logging.info('After a epoch lr:%s' % exp_lr_scheduler.get_last_lr())
            exp_lr_scheduler.step()
            if epoch % config.evaluationFrq == 0:
                # every evaluationFrq epoch to do a evaluation
                t0 = time.time()
                val_loss, val_acc, confusionMatrix = self.evaluationMode(model=model, dataloader=valid_loader,
                                                                         criterion=LossFuc, config=config)
                print('Epoch:', epoch, '| val loss: %.6f' % val_loss,
                      '| val acc: %.6f' % val_acc, '\n', '|confusion_matrix:' '\n', confusionMatrix)
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/acc', val_acc, epoch)

                t1 = time.time()
                since = t1 - t0
                logging.info('--' * 30)
                logging.info('After a evaluation lr:%s' % exp_lr_scheduler.get_last_lr())

                logging.info('epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                             % (epoch, val_loss, val_acc, since))

                # save model
                if val_acc >= config.saveValAcc and saveModelNum <= config.saveModelNumber:
                    saveModelNum += 1
                    if self.fold is not None:
                        modelFileName = 'fold%d_OCT_%s_%s_%d_[%.4f].pth' % (self.fold, config.dataSetName, config.encoder, epoch, val_acc)
                    else:
                        modelFileName = 'OCT_%s_%s_%d_[%.4f].pth' % (config.dataSetName, config.encoder, epoch, val_acc)
                    save_path = os.path.join(model_params_dir, modelFileName)
                    torch.save(model.state_dict(), save_path)
                    logging.info('saved model to %s' % save_path)
                    logging.info('--' * 30)
                # save best model
                if val_acc >= bestAcc:
                    bestAcc = val_acc
                    if self.fold is not None:
                        modelFileName = 'fold%d_OCT_%s_%s_best.pth' % (self.fold, config.dataSetName, config.encoder)
                        save_path = os.path.join(config.bestModelParamDir, modelFileName)
                        torch.save(model.state_dict(), save_path)
                        logging.info('saved best model to %s' % save_path)
                        logging.info('--' * 30)
                    else:
                        modelFileName = 'OCT_%s_%s_best.pth' % (config.dataSetName, config.encoder)
                        save_path = os.path.join(model_params_dir, modelFileName)
                        torch.save(model.state_dict(), save_path)
                        logging.info('saved best model to %s' % save_path)
                        logging.info('--' * 30)

        end = timeit.default_timer()
        print('Finish one epoch, run {} seconds'.format(end - start))

    def running(self):
        opt = self.getParser()
        logfile = '{}/{}_{}_lr_{}_{}.log'.format(opt.logfile_path, opt.dataSetName, self.timeInfo, opt.baseLR, opt.encoder)
        self.trainlog(logfile)
        model = get_encoder(opt, pretrained=True)
        if opt.teacher is not None:
            # for DeiT: use teacher to get distillation token
            teacher_encoder = getDeiT_teacher(opt, pretrained=True)
            teacher_encoder.to(opt.device)
        else:
            teacher_encoder = None
        model = model.to(opt.device)
        print('===> Loading datasets')

        if self.dataSetFlag == 'H' or self.dataSetFlag == 'V':
            trainSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=self.imgDataRoot,
                                    excelSheetName=opt.trainDataRoot, dataFlag=self.dataSetFlag, trainFlag=True)
            validSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=self.imgDataRoot,
                                    excelSheetName=opt.validDataRoot, dataFlag=self.dataSetFlag, trainFlag=False)
        else:
            if self.noise is not None:
                trainSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=opt.imgDataRoot,
                                        excelSheetName=opt.trainDataRoot, noise=self.noise)
                validSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=opt.imgDataRoot,
                                        excelSheetName=opt.validDataRoot, trainFlag=False, noise=self.noise)
            else:
                trainSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=opt.imgDataRoot,
                                        excelSheetName=opt.trainDataRoot, imgSize=self.imgSize)
                validSet = self.getData(excelFilePath=opt.trainExcelFilePath, imgRootPath=opt.imgDataRoot,
                                        excelSheetName=opt.validDataRoot, trainFlag=False, imgSize=self.imgSize)

        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_worker)
        validLoader = torch.utils.data.DataLoader(validSet, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_worker)

        optimizer = create_optimizer(opt, model)

        LossFuc = createLossFunc(opt)
        exp_lr_scheduler = create_lr_scheduler(optimizer, opt)
        import fire
        fire.Fire()
        self.train(
            model=model, teacherModel=teacher_encoder, epoch_num=self.epoch_num, start_epoch=1,
            optimizer=optimizer, LossFuc=LossFuc, exp_lr_scheduler=exp_lr_scheduler,
            train_loader=trainLoader, valid_loader=validLoader, model_params_dir=opt.model_params_dir,
            tensorboardX_dir=opt.tensorboardX_params_dir, config=opt)

