#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

from .model import BiSeNet


class FaceParser:
    def __init__(self, device="cpu"):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = torch.tensor(mapper, device=device)
        save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.pth'

        net = BiSeNet(n_classes=19)
        # Use weights_only=False for compatibility with older models
        net.load_state_dict(torch.load(save_pth, map_location=device, weights_only=False))
        self.net = net.to(device).eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0)
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)
        # Use simple indexing instead of embedding for mapping
        parsing = self.dic[parsing]
        return parsing.float()

