# -*- coding: utf-8 -*-
"""
# @Date: 2020-09-04 13:38
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: predict.py
# Copyright @ 2020 yinhao. All rights reserved.
"""

from .options.test_options import TestOptions
from .models.models import create_model
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os, time

opt = TestOptions().parse(save=False)


class PreDictor:
    def __init__(self):
        self.model = create_model(opt)
        self.model.eval()


    def get_transform(self, method=Image.BICUBIC, normalize=True):
        transform_list = []
        base = float(2 ** 4)
        base *= (2 ** 1)
        transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base, method)))
        transform_list += [transforms.ToTensor()]

        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]

        return transforms.Compose(transform_list)


    def __make_power_2(self, img, base, method=Image.BICUBIC):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)


    def tensor2im(self, image_tensor, imtype=np.uint8, normalize=True):
        if isinstance(image_tensor, list):
            image_numpy = []
            for i in range(len(image_tensor)):
                image_numpy.append(self.tensor2im(image_tensor[i], imtype, normalize))
            return image_numpy
        image_numpy = image_tensor.cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:,:,0]

        return image_numpy.astype(imtype)


    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        transform = self.get_transform()
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        generated = self.model.inference(img_tensor, None)
        img_numpy = self.tensor2im(generated.data[0])
        cartoon = Image.fromarray(img_numpy)

        return cartoon


if __name__ == "__main__":
    # 数据处理的部分
    rootdir = "/Users/yinhao/PycharmProjects/tiktok/pix2pixHD/images/whole/"
    savedir = "/Users/yinhao/PycharmProjects/tiktok/pix2pixHD/results/"
    pedictor = PreDictor()
    for img_name in os.listdir(rootdir):
        t = time.time()
        img_path = os.path.join(rootdir, img_name)
        cartoon = pedictor.predict(img_path)
        cartoon.save(savedir + "/{}".format(img_name))
        print("图片:{}, 耗时:{}".format(img_name, time.time()-t))




