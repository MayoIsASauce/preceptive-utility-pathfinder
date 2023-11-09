import cv2
import torch
import numpy as np

from time import time

from cv2.typing import MatLike

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Vision:
    def __init__(self) -> None:
        self.model_type = ["DPT_Large","DPT_Hybrid","MiDaS_small"][2]

        self.midas = torch.hub.load('intel-isl/MiDaS', self.model_type)
        self.device = torch.device('cpu')
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load('intel-isl/MiDaS', "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.vCapture = cv2.VideoCapture(0)

        self.SAFE_DEPTH: np.ndarray = None
        self.SAFE_IMAGE: str = ""
    
    def get_image(self):
        _, frame = self.vCapture.read()

        dim = None
        (h,w) = frame.shape[:2]

        width = 400
        r = width / float(w)
        dim = (width, int(h * r))

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) 
    
    def get_depth(self, img:MatLike) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction: torch.Tensor = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()

        return output

    def update(self):
        img = self.get_image()
        depth = self.get_depth(img)

        self.SAFE_DEPTH = depth
        self.SAFE_IMAGE = img