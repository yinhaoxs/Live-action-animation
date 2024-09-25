# -*- coding: utf-8 -*-
"""
# @Date: 2020-09-08 13:35
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: test_segmask.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
import argparse
import time
from torchvision import transforms
from DBFace.dbface import test
from UGATIT.utils import *
from UGATIT.networks import *
from Seg.face_seg import FaceSeg
from DBFace.models.DBFace import DBFace
from PIL import Image
from pix2pixHD.pix2pixhd import PreDictor

parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path', default="test_img/images/")
parser.add_argument('--save_path', type=str, help='cartoon save path', default="test_img/results_seg/")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)


class Photo2Cartoon:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载人脸检测模型
        self.model = DBFace()
        self.model.load("./DBFace/checkpoints/dbface.pth")
        self.model.to(self.device)

        # 加载图像分割模型
        self.segment = FaceSeg()

        # 加载卡通转换模型
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(self.device)
        params = torch.load("./UGATIT/checkpoints/douyin_jiekou.pt", map_location=self.device)
        self.genA2B.load_state_dict(params['genA2B'])
        print(" [*] Load SUCCESS")
        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def merge(self, img, A2B, axis, mask_cartoon):
        """以下代码用于贴图使用"""
        # 卡通部分
        # whole_cartoon = np.ones((img.shape[0], img.shape[1], 3))
        whole_cartoon = np.zeros((img.shape[0], img.shape[1], 3))
        whole_cartoon[axis[1]:axis[3], axis[0]:axis[2], :] = A2B/255
        whole_cartoon = whole_cartoon*255
        # # 原图部分
        whole_mask = np.zeros((img.shape[0], img.shape[1], 1))
        whole_mask[axis[1]:axis[3], axis[0]:axis[2], 0:1] = mask_cartoon
        whole_mask = 1 - whole_mask

        return whole_cartoon, whole_mask


    def inference(self, image):
        # 1.人脸检测
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        t = time.time()
        face_rgba_0, axis = test(img, self.model)
        if face_rgba_0 is None:
            print('can not detect face!!!')
            return None
        print("人脸检测的时间:{}".format(time.time()-t))

        # 2.人脸分割模型
        t = time.time()
        mask = self.segment.get_mask(face_rgba_0)
        face_rgba = np.dstack((face_rgba_0, mask))
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255).astype(np.uint8)
        face = face[..., ::-1]
        print("人像分割的时间:{}".format(time.time()-t))

        # 3.卡通脸生成
        # 采用单张图片形式
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face = self.test_transform(face)
        self.testA = (face, 0)

        self.genA2B.eval()
        with torch.no_grad():
            # 采用单张图片形式
            real_A = self.testA[0].unsqueeze(0)
            real_A = real_A.to(self.device)
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            # 生成卡通图过程
            A2B = np.array(RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))))
            A2B = A2B*255.0
            # 去除边缘卡通部分
            # A2B = (A2B * mask + 255 * (1 - mask)).astype(np.uint8)
            A2B_x = (A2B * mask).astype(np.uint8)
            A2B_x = cv2.resize(A2B_x, (face_rgba_0.shape[1], face_rgba_0.shape[0]))
            mask_whole = cv2.resize(mask, (face_rgba_0.shape[1], face_rgba_0.shape[0]))[:, : ,np.newaxis]

        print("人脸变卡通时间:{}".format(time.time()-t))

        # merge合并卡通图
        A2B_x, A2B_mask = self.merge(img, A2B_x, axis, mask_whole)

        axis.append(h)
        axis.append(w)

        return img, A2B_x, A2B_mask, axis


if __name__ == '__main__':
    args = parser.parse_args()
    c2p = Photo2Cartoon()
    pix2pix = PreDictor()
    num = 0
    for photo_name in os.listdir(args.photo_path):
        num += 1
        t = time.time()
        img_path = os.path.join(args.photo_path+os.sep, photo_name)
        # pix2pixHD算法预测全图卡通
        whole = pix2pix.predict(img_path)
        # 将全图卡通从pillow转成opencv
        whole = cv2.cvtColor(np.asarray(whole), cv2.COLOR_RGB2BGR)
        # 生成人脸卡通A2B, 人脸部分mask
        img, A2B, A2B_mask, axis = c2p.inference(img_path)
        # 将全图卡通resize成与原图相同的形状
        wholecartoon = cv2.resize(whole, (img.shape[1], img.shape[0]))

        # 相乘
        out = (wholecartoon*A2B_mask).astype(np.uint8) + A2B
        if out is not None:
            cv2.imwrite(args.save_path+"/"+"{}".format(photo_name), out)

        print("单张图片的处理时间:{}".format(time.time()-t)+'\n')
