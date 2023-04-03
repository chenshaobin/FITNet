import itertools
import matplotlib.pyplot as plt
from typing import List
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import torch
import os
from sklearn.manifold import TSNE

colorList = ['g-', 'y-', 'r-', 'darkblue', 'black', 'gold', 'violet',  '#d1b26f']
scatterColorList = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
colorList_PM_PS = {'ViT': '#95d0fc',
                   'ResNet50': '#029386',
                   'CrossViT-Ti': '#f97306',
                   'DeiT': '#607c8e',
                   'E-Net': '#eedc5b',
                   'Swin-T': '#929591',
                   'ConvNeXt-T': '#89fe05',
                   'FIT-Net': '#e50000'}

def plot_confusion_matrix(cm, classes, savePath: str, normalize=False,
                          title=None, cmap=plt.cm.Paired):
    # plt.cm.Blues

    font1 = {'family': 'sans-serif', 'weight': 'normal', 'size': 16}
    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # cm = cm*100
    print(cm)
    plt.figure()
    img = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar(img)
    plt.colorbar(img, fraction=0.046, pad=0.04)
    # plt.title('Accuracy = {:.2f}%'.format(accu), font1)
    plt.title(title, fontsize=10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=12)
    plt.yticks(tick_marks, classes, rotation=90, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontdict=font1)

    # cbar = plt.colorbar(plt.imshow(cm, interpolation='nearest', cmap=cmap))
    # cbar.set_clim(vmin=0, vmax=100)

    # plt.ylabel('True label', fontsize=16)
    # plt.xlabel('Predicted label', verticalalignment='top', fontsize=16)
    plt.tight_layout()
    plt.savefig(savePath, bbox_inches='tight', dpi=600)
    plt.clf()

def metric_results(pred_label, gt_label, scores):
    # Calculate some evaluation criteria
    """
                N      P  -> pre
     |      N   TN     FP
    real    P   FN     TP
    """
    result = {}
    cm = confusion_matrix(gt_label, pred_label)
    result.update({'confusion_matrix': cm})

    accuracy = accuracy_score(gt_label, pred_label)
    result.update({'accuracy': accuracy})

    AUC = roc_auc_score(np.array(gt_label), np.array(scores)[:, 1])
    result.update({'AUC': AUC})

    precision = precision_score(gt_label, pred_label)
    result.update({'precision': precision})
    # precision_1 = (cm[1][1]) / (cm[1][1] + cm[0][1])
    # print('precision_1"', precision_1)
    recall = recall_score(gt_label, pred_label)
    result.update({'sensitivity': recall})  # recall和sensitivity计算方法一样
    # recall_ = (cm[1][1]) / (cm[1][1] + cm[1][0])
    # print('sensitivity_:', recall_)
    F1_score = f1_score(gt_label, pred_label)
    result.update({'F1_score': F1_score})

    # calculate specificity, For 2 class
    specificity = (cm[0][0]) / (cm[0][0] + cm[0][1])
    result.update({'specificity': specificity})

    kappa = cohen_kappa_score(gt_label, pred_label)
    result.update({'Kappa': kappa})
    return result

def metric_results_multiClass(classNumber: int, pred_label, gt_label, scores, metricAverage="macro"):
    # metricAverage="weighted" or metricAverage="macro"
    result = {}
    cm = confusion_matrix(gt_label, pred_label)
    result.update({'confusion_matrix': cm})

    accuracy = accuracy_score(gt_label, pred_label)
    result.update({'accuracy': accuracy})
    # multi_class='ovo'
    AUC = roc_auc_score(np.array(gt_label), np.array(scores), multi_class='ovr', average=metricAverage)
    result.update({'AUC': AUC})

    precision = precision_score(gt_label, pred_label, average=metricAverage)
    result.update({'precision': precision})

    recall = recall_score(gt_label, pred_label, average=metricAverage)
    result.update({'recall': recall})

    F1_score = f1_score(gt_label, pred_label, average=metricAverage)
    result.update({'F1_score': F1_score})
    # calculate specificity
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    TNR = TN / (TN + FP)  # specificity
    # print('specificity:', TNR)
    result.update({'specificity': np.sum(TNR) / classNumber})
    kappa = cohen_kappa_score(gt_label, pred_label)
    result.update({'Kappa': kappa})
    return result

def metric_result_for_eachClass(multiClassConfusionMatrix: np.ndarray) -> List:
    # In the multi-classification task, the classification index of each category is calculated separately according to the confusion matrix.
    def metric_results(TP, TN, FN, FP):
        result = {}

        accuracy = (TP + TN) / (TP + TN + FN + FP)
        result.update({'accuracy': accuracy})

        precision = TP / (TP + FP)
        result.update({'precision': precision})

        recall = TP / (TP + FN)
        result.update({'sensitivity': recall})  # recall和sensitivity计算方法一样

        F1_score = 1 / (1/2 * (1/precision + 1/recall))
        result.update({'F1_score': F1_score})

        # calculate specificity, For 2 class
        specificity = TN / (TN + FP)
        result.update({'specificity': specificity})

        return result

    dataClass = multiClassConfusionMatrix.shape[0]
    result = []

    for labelIndex in range(dataClass):
        ALL = multiClassConfusionMatrix.sum()
        TP = multiClassConfusionMatrix[labelIndex][labelIndex]
        # Each column of the confusion matrix corresponds to the total number of predicted classes
        FP = np.sum(multiClassConfusionMatrix[:, labelIndex]) - TP
        # The sum of each row represents the real number of corresponding categories
        FN = np.sum(multiClassConfusionMatrix[labelIndex, :]) - TP
        TN = ALL - TP - FP - FN
        print(f"TP:{TP}, TN:{TN},FN:{FN},FP:{FP}")
        assert TP + TN + FN + FP == ALL, "size of TP + TN + FN + FP and Total is not equal!"
        metricResultDict = metric_results(TP, TN, FN, FP)
        # printMetricResults(metricResultDict)
        result.append(metricResultDict)
    return result

def printMetricResults(myDict):
    for item in myDict:
        print(item, ":", myDict[item])

def plotRocCurve(true_label, scores, save_path, picName, picLabel='PMCNV'):
    plt.rcParams['figure.figsize'] = (5.2, 4.6)
    font1 = {'family': 'sans-serif', 'weight': 'normal', 'size': 8}
    scores = np.array(scores)
    fpr, tpr, _ = roc_curve(np.array(true_label), scores[:, 1])
    roc_auc = auc(fpr, tpr)

    print('RocCurve AUC:', roc_auc)

    color_index = ['g-']
    # PMCNV, Macular Hole, Retinal Detachment, Retinoschisis
    label_index = ['{} (AUC = {:.4f})']
    f = plt.figure(1)
    plt.plot(fpr, tpr, color_index[0], lw=2,
             label=label_index[0].format(picLabel, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.title('ROC of {} '.format(picLabel), fontsize=6)
    plt.legend(loc="lower right", prop=font1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, picName), dpi=600)
    # plt.show()
    plt.clf()

def plotRocCurve_multiClass(classNumber: int, true_label, scores, save_path, picName, picLabel: List):
    font1 = {'family': 'sans-serif', 'weight': 'normal', 'size': 8}
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_label = np.arange(classNumber)
    print('class_label:', class_label)
    # Binarize the label
    true_label = label_binarize(true_label, classes=class_label)
    scores = np.array(scores)
    # One vs rest method to calculate TPR/FPR and AUC for each category
    for i in range(classNumber):
        fpr[i], tpr[i], _ = roc_curve(np.array(true_label[:, i]), scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    color_index = []
    label_index = []
    for i in range(classNumber):
        label_index.append('{} (AUC = {:.4f})')
        color_index.append(colorList[i])

    f = plt.figure(1)
    for i in range(len(class_label)):
        plt.plot(fpr[i], tpr[i], color_index[i], lw=2,
                 label=label_index[i].format(picLabel[i], roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.title('ROC Curve', fontsize=6)
    plt.legend(loc="lower right", prop=font1)
    #  [left, bottom, width, height]
    """
    insert_axes = f.add_axes([0.5, 0.52, 0.4, 0.4])
    for i in range(classNumber):
        insert_axes.plot(fpr[i], tpr[i], color_index[i], lw=2, label=label_index[i].format(picLabel[i], roc_auc[i]))
        insert_axes.set_xlim([-0.07, 0.3])
        insert_axes.set_ylim([0.6, 1.01])
        insert_axes.get_yaxis().set_visible(False)
        insert_axes.get_xaxis().set_visible(False)
    """
    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, picName), dpi=600)
    # plt.show()
    plt.clf()
