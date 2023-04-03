import os.path
from os.path import join
import torch
from typing import List
from EvaluateMetrics.pytorch_GradCam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,  XGradCAM, EigenCAM, EigenGradCAM,  LayerCAM, FullGrad
from EvaluateMetrics.pytorch_GradCam.utils.image import show_cam_on_image, deprocess_image, preprocess_image, preprocess_image_batch

import numpy as np
import cv2


class getGramCam:
    def __init__(self, model, target_layers, imgSize: int, img_mean: List,
                 img_std: List, savePath: str, device, method='gradcam',
                 batchSize=1, vitFlag=False):
        # Choose the target layer you want to compute the visualization for.
        # Usually this will be the last convolutional layer in the model.
        # Some common choices can be:
        # Resnet18 and 50: model.layer4[-1]
        # VGG, densenet161: model.features[-1]
        # mnasnet1_0: model.layers[-1]
        # You can print the model to help chose the layer
        # You can pass a list with several target layers,
        # in that case the CAMs will be computed per layer and then aggregated.
        # You can also try selecting all layers of a certain type, with e.g:
        # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
        # find_layer_types_recursive(model, [torch.nn.ReLU])
        self.device = device
        self.model = model
        self.method = method
        self.imgSize = imgSize
        self.img_mean = img_mean
        self.img_std = img_std
        self.savePath = savePath
        self.batchSize = batchSize
        self.vitFlag = vitFlag
        if target_layers is None:
            self.target_layers = [self.model.layer4[-1]]
        else:
            self.target_layers = target_layers

        self.methods = \
            {"gradcam": GradCAM,
             "scorecam": ScoreCAM,
             "gradcam++": GradCAMPlusPlus,
             "ablationcam": AblationCAM,
             "xgradcam": XGradCAM,
             "eigencam": EigenCAM,
             "eigengradcam": EigenGradCAM,
             "layercam": LayerCAM,
             "fullgrad": FullGrad}
        self.aug_smooth = False
        self.eigen_smooth = False

    def reshape_transform_ViT(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                          height, width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def reshape_transform_SwinT_s(self, tensor, height=7, width=7):
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        # print(f"before reshape:{tensor.shape}, reshape:{result.shape}")
        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        # print(f"reshape output:{result.shape}")
        return result

    def reshape_transform_SwinT_h(self, tensor, height=4, width=4):
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        # print(f"before reshape:{tensor.shape}, reshape:{result.shape}")
        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        # print(f"reshape output:{result.shape}")
        return result

    def mask2box(self, img, mask):
        mask = 255 * mask
        img = img * 255
        # plt.subplot(1,3,1), plt.imshow(mask)
        # (src, thresh, maxval, type), cv2.THRESH_BINARY--->小于阈值的像素值置0
        # ret, thresh = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        mask = mask.astype("uint8")
        ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Optimal threshold:{ret}!")
        # plt.subplot(1,3, 2), plt.imshow(thresh)
        thresh = np.array(thresh, np.uint8)
        contours, _2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        # 0: draw the contour of max area, -1: draw all contour
        cv2.drawContours(img, contours, 0, (0, 255, 0), 3, lineType=cv2.LINE_AA)
        # draw rectangle according to contours
        """
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        """
        # plt.subplot(1,3, 3), plt.imshow(img)
        # cv2.imwrite(save_path, img)
        return img

    def getImg(self, imgPath):
        rgb_img = cv2.resize(cv2.imread(imgPath, 1), (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        rgb_img_original = np.float32(cv2.resize(cv2.imread(imgPath, 1), (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA))/255
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=self.img_mean, std=self.img_std)
        print('img shape:{}, preprocess_image shape:{}'.format(rgb_img.shape, input_tensor.shape))

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        target_category = None
        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = self.methods[self.method]
        cam_image = None
        if not self.vitFlag:
            print(f"Processing!")
            with cam_algorithm(model=self.model,
                               target_layers=self.target_layers,
                               device=self.device,
                               use_cuda=True) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = self.batchSize
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=self.aug_smooth,
                                    eigen_smooth=self.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                print('original grayscale_cam shape:', grayscale_cam.shape)
                grayscale_cam = grayscale_cam[0, :]
                print('grayscale_cam shape:', grayscale_cam.shape)

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        else:
            with cam_algorithm(model=self.model,
                               target_layers=self.target_layers,
                               device=self.device,
                               use_cuda=True,
                               # [self.reshape_transform_SwinT_s, self.reshape_transform_SwinT_h]
                               reshape_transform=[self.reshape_transform_SwinT_s]) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = self.batchSize
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=self.aug_smooth,
                                    eigen_smooth=self.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                print('original grayscale_cam shape:', grayscale_cam.shape)
                grayscale_cam = grayscale_cam[0, :]
                print('grayscale_cam shape:', grayscale_cam.shape)
                mask_to_box = self.mask2box(rgb_img_original, grayscale_cam)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        camSavePath = join(self.savePath, 'CamImg')
        if not os.path.isdir(camSavePath):
            os.makedirs(camSavePath)
        maskToBoxPath = join(self.savePath, 'Mask2Box')
        if not os.path.isdir(maskToBoxPath):
            os.makedirs(maskToBoxPath)

        camSavePath = join(camSavePath, imgPath.split('/')[-1])
        print(f"GradCam save img:{camSavePath}")
        cv2.imwrite(camSavePath, cam_image)

        maskToBoxPath = join(maskToBoxPath, imgPath.split('/')[-1])
        print(f"mask2Box save img:{maskToBoxPath}")
        cv2.imwrite(maskToBoxPath, mask_to_box)

    def getImgBatch(self, imgPathBatch):
        rgb_imgBatch = np.ones([len(imgPathBatch), 224, 224, 3])
        rgb_img_orginalBatch = np.ones([len(imgPathBatch), 224, 224, 3])
        for index, imgPath in enumerate(imgPathBatch):
            rgb_img = cv2.resize(cv2.imread(imgPath, 1), (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            rgb_img_original = np.float32(cv2.resize(cv2.imread(imgPath, 1), (self.imgSize, self.imgSize), interpolation=cv2.INTER_AREA))/255
            rgb_img_orginalBatch[index, :, :, :] = rgb_img_original
            rgb_img = np.float32(rgb_img) / 255
            rgb_imgBatch[index, :, :, :] = rgb_img

        input_tensor = preprocess_image_batch(rgb_imgBatch, mean=self.img_mean, std=self.img_std)
        print('rgb_imgBatch shape:{}, preprocess_image shape:{}'.format(rgb_imgBatch.shape, input_tensor.shape))

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        target_category = None
        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = self.methods[self.method]
        mask_to_boxList = []
        cam_imageList = []
        if not self.vitFlag:
            print(f"Processing!")
            with cam_algorithm(model=self.model,
                               target_layers=self.target_layers,
                               device=self.device,
                               use_cuda=True) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = self.batchSize
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=self.aug_smooth,
                                    eigen_smooth=self.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                print('original grayscale_cam shape:', grayscale_cam.shape)
                grayscale_cam = grayscale_cam[0, :]
                print('grayscale_cam shape:', grayscale_cam.shape)

                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        else:
            with cam_algorithm(model=self.model,
                               target_layers=self.target_layers,
                               device=self.device,
                               use_cuda=True,
                               # [self.reshape_transform_SwinT_s, self.reshape_transform_SwinT_h], [self.reshape_transform_SwinT_s]
                               reshape_transform=[self.reshape_transform_SwinT_s, self.reshape_transform_SwinT_h]) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = self.batchSize
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=self.aug_smooth,
                                    eigen_smooth=self.eigen_smooth)

                print('original grayscale_cam shape:', grayscale_cam.shape)
                for index in range(grayscale_cam.shape[0]):
                    grayscaleCam = grayscale_cam[index, :, :]
                    print('grayscaleCam shape:', grayscaleCam.shape)
                    mask_to_box = self.mask2box(rgb_img_orginalBatch[index, :, :, :], grayscaleCam)
                    cam_image = show_cam_on_image(rgb_imgBatch[index, :, :, :], grayscaleCam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    mask_to_boxList.append(mask_to_box)
                    cam_imageList.append(cam_image)

        camSavePath = join(self.savePath, 'CamImg_twoLayer')
        if not os.path.isdir(camSavePath):
            os.makedirs(camSavePath)
        maskToBoxPath = join(self.savePath, 'Mask2Box_twoLayer')
        if not os.path.isdir(maskToBoxPath):
            os.makedirs(maskToBoxPath)
        print(f"mask_to_boxList:{len(mask_to_boxList)}, cam_imageList:{len(cam_imageList)}")
        for index, imgPath in enumerate(imgPathBatch):
            camImgSavePath = join(camSavePath, imgPath.split('/')[-1])
            print(f"GradCam save img:{camImgSavePath}")
            cv2.imwrite(camImgSavePath, cam_imageList[index])

            maskImgToBoxPath = join(maskToBoxPath, imgPath.split('/')[-1])
            print(f"mask2Box save img:{maskImgToBoxPath}")
            cv2.imwrite(maskImgToBoxPath, mask_to_boxList[index])
