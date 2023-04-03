import os
import torch
import torch.optim as optim
import math
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import mmcv
import torch.nn.functional as F
from scipy.stats import pearsonr
import datetime
import time
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import defaultdict, deque
import torch.distributed as dist
from typing import List
# from torchvision import transforms
from PIL import Image
from torchvision import transforms as T
from data.imageDataTransform import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize_and_RandomCrop, RandomCrop, ToTensor
from torchvision.utils import make_grid
from torch.autograd import Variable
from os.path import join
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def normalizeImg(imgTensor):
    if isinstance(imgTensor, list):
        imgTensor = torch.stack(imgTensor, dim=0)

    if imgTensor.dim() == 2:  # single image H x W
        imgTensor = imgTensor.unsqueeze(0)
    if imgTensor.dim() == 3:  # single image
        if imgTensor.size(0) == 1:  # if single-channel, convert to 3-channel
            imgTensor = torch.cat((imgTensor, imgTensor, imgTensor), 0)
        imgTensor = imgTensor.unsqueeze(0)

    if imgTensor.dim() == 4 and imgTensor.size(1) == 1:  # single-channel images
        imgTensor = torch.cat((imgTensor, imgTensor, imgTensor), 1)

    imgTensor = imgTensor.clone()  # avoid modifying tensor in-place

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
        return img

    imgTensorNormalize = norm_ip(imgTensor, float(imgTensor.min()), float(imgTensor.max()))
    return imgTensorNormalize

def TensorToPILImage(imageTensor):
    imageTensor = normalizeImg(imageTensor)
    image = imageTensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def imgTensorShow(imageTensor, title):
    imageTensor = normalizeImg(imageTensor)
    image = imageTensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imgShow(pilImgFile, imgType='RGB', title=None):
    if imgType == 'gray':
        plt.imshow(pilImgFile, cmap ='gray')
    else:
        plt.imshow(pilImgFile)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show(imgs):
    import torchvision.transforms.functional as F
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """
        Track a series of values and provide access to smoothed values over a
        window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def create_optimizer(config, model):
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.baseLR, betas=(0.9, 0.999))
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.baseLR, momentum=0.9, nesterov=True)
    else:
        raise ValueError("No such optimizer: {}".format(config.optimizer))

    return optimizer

def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cosineAnnealing':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.warmup_epochs, eta_min=config.min_lr)
    elif config.lr_scheduler == 'step':
        # For each step_size epoch, the learning rate is multiplied by gamma to perform the attenuation operation
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.warmup_epochs, gamma=0.9)
    elif config.lr_scheduler == 'multistep':
        if config.steps is None: return None
        if isinstance(config.steps, int): config.steps = [config.steps]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = lr_scheduler.exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'consine':
        # cosine
        lf = lambda x: ((1 + math.cos(x * math.pi / config.warmup_epochs)) / 2) * (1 - config.lrf) + config.lrf
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif config.lr_scheduler == 'none':
        scheduler = None
    else:
        raise ValueError("No such scheduler: {}".format(config.lr_scheduler))
    return scheduler


def createLossFunc(config):
    if config.lossName == 'crossEntropyLoss':
        return CrossEntropyLoss()
    elif config.lossName == 'focalLoss':
        return FocalLoss(class_num=config.classNumber, gamma=config.focalLossGamma)
    else:
        raise ValueError("No such LossFunction: {}".format(config.lossName))

def get_mean_std(data: List) -> dict:
    dataMean = np.mean(data)
    dataStd = np.std(data)
    dataMedian = np.median(data)
    return {'mean': dataMean, 'std': dataStd, 'median': dataMedian}

def get_PearsonCorrelation(data1: List, data2: List):
    pearsonCorrelationCoefficient, pValue = pearsonr(data1, data2)
    return pearsonCorrelationCoefficient, pValue

def create_lr_scheduler_step(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Returns a learning rate multiplier based on the number of steps
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # During the warmup process, the lr multiplication factor changes from warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def saveDataAs_npz(fileName: str, data: List):
    # for key, value in data:
    np.savez(fileName, data=data)

def read_npz_data(fileName: str):
    data = np.load(fileName, allow_pickle=True)
    return data

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        eps = 1e-7
        device = inputs.device
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        # print(f"ids:{ids}")
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        alpha = alpha.to(device)
        # print(f"alpha:{alpha}")

        probs = (P * class_mask).sum(1).view(-1, 1)
        probs = probs.to(device)
        # print(f"probs:{probs}, probs + eps:{probs + eps}")
        log_p = torch.log(probs + eps)
        log_p = log_p.to(device)
        # log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Get_mean_std_for_dataset:
    def __init__(self, dataRootPath:str):
        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        self.transform = transforms.Compose([transforms.ToTensor()])
        # Datasets are already stored in different folders by category
        self.dataset = ImageFolder(dataRootPath, self.transform)

    def get_mean_std(self, mean_std_path=None):
        """
            Calculate the mean and standard deviation of a dataset
            :param mean_std_path: The file path where the calculated mean and standard deviation are stored
        """
        num_imgs = len(self.dataset)
        for data in self.dataset:
            img = data[0]
            for i in range(3):
                # calculate for every channel
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        # Write the resulting mean and standard deviation to a file, which can then be read from
        """
        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stdevs, f)
            print('pickle done')
        """
        return self.means, self.stdevs