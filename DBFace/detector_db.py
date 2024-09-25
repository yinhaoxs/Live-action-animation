# -*- coding: utf-8 -*-
"""
# @Date: 2020-06-05 17:16
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: test.py
# Copyright @ 2020 yinhao. All rights reserved.
"""

from .models import common
from .models.DBFace import DBFace
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .models.common import intv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs
    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue
        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]
    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if torch.cuda.is_available():
        torch_image = torch_image.cuda()

    with torch.no_grad():
        model.eval()
        hm, box, landmark = model(torch_image)
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
        hm_height, hm_width = hm.shape[2:]

        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices // hm_width).int().data.numpy())
        xs = list((indices % hm_width).int().data.numpy())
        scores = list(scores.data.numpy())
        box = box.cpu().squeeze().data.numpy()
        landmark = landmark.cpu().squeeze().data.numpy()

        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[:, cy, cx]
            x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def detect_faces(image, model):
    # ### 定义初始化人脸检测模型
    # model = DBFace()
    # model.load("./DBFace/checkpoints/dbface.pth")
    # if torch.cuda.is_available():
    #     model.cuda()

    ### 检测人脸与关键点
    bounding_boxes, landmarks = [], []
    objs = detect(model, image)
    for obj in objs:
        x, y, r, b = intv(obj.box)
        bounding_boxes.append([float(x), float(y), float(r), float(b)])
        ### 获取dbface模型的landmark点
        if obj.haslandmark:
            for i in range(len(obj.landmark)):
                x, y = obj.landmark[i][:2]
                landmarks.append([float(x), float(y)])

    return np.array(bounding_boxes), np.array(landmarks)


if __name__ == "__main__":
    pass