# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target=None):
        size = self.get_size(images[0].size)

        for i, _ in enumerate(images):
            images[i] = F.resize(images[i], size)

        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return images
        else:
            target = target.resize(size)
        return images, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target):
        if random.random() < self.prob:
            for i, _ in enumerate(images):
                images[i] = F.hflip(images[i])
            target = target.transpose(0)
        return images, target


class ConcatImages(object):
    def __call__(self, images, target):
        for i, _ in enumerate(images):
            images[i] = np.array(images[i])

        concat_image = images[0]
        for image in images[1:]:
            if image.ndim == 2:
                image = image[:, :, np.newaxis]

            concat_image = np.concatenate((concat_image, image), axis=2)

        return concat_image, target


class ToTensor(object):
    def __call__(self, image, target):
        print(F.to_tensor(image).shape)
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target