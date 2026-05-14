#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# This code accompanies the research paper: Upchurch, Paul, and Ransen
# Niu. "A Dense Material Segmentation Dataset for Indoor and Outdoor
# Scene Parsing." ECCV 2022.
#
# This example shows how to predict materials.
#

from collections import Counter
import torchvision.transforms as TTR
import os
import random
import json
import cv2
import numpy as np
import torch
import math
from PIL import Image

random.seed(112)

dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23,
    24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49,
    50, 51, 52, 53, 56]

t = json.load(open(os.path.expanduser('demo/taxonomy.json'), 'rb'))

namemap = [
    t['names'][i] for i in range(len(t['names'])) if i in dms46
]

namemap = np.array(namemap)


def find_commons(lst):
    counter = Counter(lst)
    commons = counter.most_common(2)

    if len(commons) == 1:
        return commons[0][0], 'None'

    return commons[0][0], commons[1][0]


def apply_name(label_mask):
    # translate labels to visualization colors
    vis = np.take(namemap, label_mask, axis=0)
    return vis[..., ::-1]


def find_material(img):
    is_cuda = torch.cuda.is_available()
    model = torch.jit.load('demo/DMS46_v1.pt')

    if is_cuda:
        model = model.cuda()

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    new_dim = 512
    h, w = img.shape[0:2]

    scale_x = float(new_dim) / float(h)
    scale_y = float(new_dim) / float(w)
    scale = min(scale_x, scale_y)
    new_h = math.ceil(scale * h)
    new_w = math.ceil(scale * w)

    img = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    img = np.array(img)

    image = torch.from_numpy(img.transpose((2, 0, 1))).float()
    image = TTR.Normalize(mean, std)(image)

    if is_cuda:
        image = image.cuda()

    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)[0].data.cpu()[0, 0].numpy()

    predicted_names = apply_name(prediction)

    return predicted_names
