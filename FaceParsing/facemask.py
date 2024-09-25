#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 255, 255], [0, 0, 0], [0, 0, 0], # [], 面部, 右眉毛
                   [0, 0, 0], [0, 0, 0], # 左眉毛, 右眼
                   [0, 0, 0], [0, 0, 0], [255, 255, 255], # 左眼, eye_g, 右耳
                   [255, 255, 255], [255, 255, 255],  # ear_r, 左耳
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],  # 鼻子， 嘴巴，上嘴唇，
                   [0, 0, 0], [255, 255, 255],  # 下嘴唇，脖子，
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], # [], cloth, hair
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], # 帽子
                   [255, 255, 255], [255, 255, 255], [255, 255, 255]]

    # part_colors = [[255, 255, 255], [0, 0, 0], [0, 0, 0], # [], 面部, 右眉毛
    #                [0, 0, 0], [0, 0, 0], # 左眉毛, 右眼
    #                [0, 0, 0], [0, 0, 0], [255, 255, 255], # 左眼, eye_g, 右耳
    #                [255, 255, 255], [255, 255, 255],  # ear_r, 左耳
    #                [0, 0, 0], [0, 0, 0], [0, 0, 0],  # 鼻子， 嘴巴，上嘴唇，
    #                [0, 0, 0], [255, 255, 255],  # 下嘴唇，脖子，
    #                [255, 255, 255], [255, 255, 255], [0, 0, 0], # [], cloth, hair
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], # 帽子
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255]]

    # part_colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], # [], 面部, 右眉毛
    #                [255, 255, 255], [255, 255, 255], # 左眉毛, 右眼
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], # 左眼, eye_g, 右耳
    #                [255, 255, 255], [255, 255, 255],  # ear_r, 左耳
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255],  # 鼻子， 嘴巴，上嘴唇，
    #                [255, 255, 255], [255, 255, 255],  # 下嘴唇，脖子，
    #                [255, 255, 255], [255, 255, 255], [0, 0, 0], # [], cloth, hair
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255], # 帽子
    #                [255, 255, 255], [255, 255, 255], [255, 255, 255]]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0, vis_parsing_anno_color, 1, 0)

    return vis_im


def evaluate(img, net):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    net.eval()
    with torch.no_grad():
        # img = Image.open(img_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print("#####", len(np.unique(parsing)))
        vis_im = vis_parsing_maps(image, parsing, stride=1)

    return vis_im



if __name__ == "__main__":
    # evaluate(dspth='test-img/', cp='79999_iter.pth')
    img_dir = 'test-img/'
    res_path = 'test-res/'
    for img_name in os.listdir(img_dir):
        img_path = osp.join(img_dir, img_name)
        vis_im = evaluate(img_path)
        cv2.imwrite(res_path+"/{}.jpg".format(img_name.split(".")[0]), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])





