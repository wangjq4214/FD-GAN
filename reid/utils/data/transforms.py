import random
import math
import numpy as np

from PIL import Image

# 这个写法可以学习, python 函数柯里化的一个写法


class RectScale:
    """
    图片 resize
    """

    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop:
    """
    图片随机裁剪和 resize
    """

    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for _ in range(10):
            # 进行尝试, 可能不成功
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # 如果不成功直接 resize
        scale = RectScale(self.height, self.width, self.interpolation)
        return scale(img)


class RandomSizedEarser:
    """
    随机区域擦除
    """

    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.5):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1.0, 1.0)
        w, h = img.size
        area = w * h

        if p1 > self.p:
            return img

        gen = True
        while gen:
            se = random.uniform(self.sl, self.sh) * area
            re = random.uniform(self.asratio, 1/self.asratio)
            he = np.sqrt(se*re)
            we = np.sqrt(se/re)
            xe = random.uniform(0, w-we)
            ye = random.uniform(0, h-he)
            if se+we <= w and ye+he <= h and xe > 0 and ye > 0:
                x1 = int(np.ceil(xe))
                y1 = int(np.ceil(ye))
                x2 = int(np.floor(x1+we))
                y2 = int(np.floor(y1+he))
                part1 = img.crop((x1, y1, x2, y2))
                rc = random.randint(0, 255)
                gc = random.randint(0, 255)
                bc = random.randint(0, 255)
                I = Image.new('RGB', part1.size, (rc, gc, bc))
                img.paste(I, part1.size)

                return img
